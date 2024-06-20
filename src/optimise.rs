fn save(tm: TMClassifier, filepath: &str) {
    let filepath = if !filepath.ends_with(".tm") {
        format!("{}.tm", filepath)
    } else {
        filepath.to_string()
    };

    println!("Saving model to {}... ", filepath);
    let classifier_str = serde_json::to_string(&tm).unwrap();
    std::fs::write(filepath, classifier_str).unwrap();
    println!("Done.\n");
}

fn load(filepath: &str) -> TMClassifier {
    let filepath = if !filepath.ends_with(".tm") {
        format!("{}.tm", filepath)
    } else {
        filepath.to_string()
    };

    println!("Loading model from {}... ", filepath);
    let classifier_str = std::fs::read_to_string(filepath).unwrap();
    let tm: TMClassifier = serde_json::from_str(&classifier_str).unwrap();
    println!("Done.\n");

    tm
}

fn mix(clauses_vec: Vec<Array2<u8>>) -> Array2<u8> {
    let mut result = Array2::from_elem(clauses_vec[0].dim(), 0);
    for (i, mut c) in result.outer_iter_mut().enumerate() {
        for j in 0..c.len() {
            c[j] = clauses_vec
                .iter()
                .map(|clauses| clauses[(i, j)])
                .max()
                .unwrap();
        }
    }

    result
}

#[derive(Debug)]
enum Algorithm {
    Merge,
    Join,
}

fn merge(new_tm: &mut TMClassifier, tms: &[TMClassifier], algo: Option<&Algorithm>) {
    let algo = algo.unwrap_or(&Algorithm::Merge);
    for cls in 0..new_tm.clauses.len() {
        match algo {
            Algorithm::Merge => {
                new_tm.clauses.get_mut(cls).unwrap().positive_clauses = mix(tms
                    .iter()
                    .map(|tm| tm.clauses[cls].positive_clauses.clone())
                    .collect());
                new_tm.clauses.get_mut(cls).unwrap().negative_clauses = mix(tms
                    .iter()
                    .map(|tm| tm.clauses[cls].negative_clauses.clone())
                    .collect());
            }
            Algorithm::Join => {
                let positive: Vec<ArrayBase<ViewRepr<&u8>, Dim<[usize; 2]>>> = tms
                    .iter()
                    .map(|tm| tm.clauses[cls].positive_clauses.view())
                    .collect::<Vec<_>>();

                let negative = tms
                    .iter()
                    .map(|tm| tm.clauses[cls].negative_clauses.view())
                    .collect::<Vec<_>>();

                new_tm.clauses.get_mut(cls).unwrap().positive_clauses = hcat(&positive, &negative);
                new_tm.clauses.get_mut(cls).unwrap().negative_clauses = hcat(&negative, &positive);

                let clauses_num_half = new_tm.clauses[cls].positive_clauses.ncols() as i64;
                new_tm.clauses_num = clauses_num_half * 2;

                new_tm
                    .clauses
                    .get_mut(cls)
                    .unwrap()
                    .positive_included_literals = vec![vec![]; clauses_num_half as usize];
                new_tm
                    .clauses
                    .get_mut(cls)
                    .unwrap()
                    .negative_included_literals = vec![vec![]; clauses_num_half as usize];
            }
        }

        for (j, c) in new_tm.clone().clauses[cls]
            .positive_clauses
            .outer_iter()
            .enumerate()
        {
            new_tm
                .clauses
                .get_mut(cls)
                .unwrap()
                .positive_included_literals[j] = c
                .iter()
                .enumerate()
                .filter_map(|(i, &value)| {
                    if value >= new_tm.clauses[cls].include_limit {
                        Some(i as u16)
                    } else {
                        None
                    }
                })
                .collect();
        }

        for (j, c) in new_tm.clone().clauses[cls]
            .negative_clauses
            .outer_iter()
            .enumerate()
        {
            new_tm
                .clauses
                .get_mut(cls)
                .unwrap()
                .negative_included_literals[j] = c
                .iter()
                .enumerate()
                .filter_map(|(i, &value)| {
                    if value >= new_tm.clauses[cls].include_limit {
                        Some(i as u16)
                    } else {
                        None
                    }
                })
                .collect();
        }
    }
}

fn hcat(
    first: &[ArrayBase<ViewRepr<&u8>, Dim<[usize; 2]>>],
    second: &[ArrayBase<ViewRepr<&u8>, Dim<[usize; 2]>>],
) -> Array2<u8> {
    let positive = concatenate(Axis(1), first).unwrap();
    let negative = concatenate(Axis(1), second).unwrap();
    concatenate![Axis(1), positive, negative]
}

fn merge_clauses(new_tm: &mut TMClassifierCompiled, tms: &[TMClassifierCompiled], algo: Algorithm) {
    let num_clauses = new_tm.clauses.len();
    for cls in (0..num_clauses) {
        match algo {
            Algorithm::Merge => {
                let mut new_positive = Vec::new();
                let mut new_negative = Vec::new();

                for tm in tms {
                    if let Some(clause) = tm.clauses.get(cls) {
                        for (i, pos_literals) in
                            clause.positive_included_literals.iter().enumerate()
                        {
                            if i >= new_positive.len() {
                                new_positive.push(HashSet::new());
                            }
                            for &lit in pos_literals {
                                new_positive[i].insert(lit);
                            }
                        }
                        for (i, neg_literals) in
                            clause.negative_included_literals.iter().enumerate()
                        {
                            if i >= new_negative.len() {
                                new_negative.push(HashSet::new());
                            }
                            for &lit in neg_literals {
                                new_negative[i].insert(lit);
                            }
                        }
                    }
                }

                if let Some(clause) = new_tm.clauses.get_mut(cls) {
                    clause.positive_included_literals = new_positive
                        .into_iter()
                        .map(|set| set.into_iter().collect())
                        .collect();
                    clause.negative_included_literals = new_negative
                        .into_iter()
                        .map(|set| set.into_iter().collect())
                        .collect();
                }
            }
            Algorithm::Join => {
                let mut all_positive = HashSet::new();
                let mut all_negative = HashSet::new();

                for tm in tms {
                    if let Some(clause) = tm.clauses.get(cls) {
                        for pos_literals in &clause.positive_included_literals {
                            for &lit in pos_literals {
                                all_positive.insert(lit);
                            }
                        }
                        for neg_literals in &clause.negative_included_literals {
                            for &lit in neg_literals {
                                all_negative.insert(lit);
                            }
                        }
                    }
                }

                if let Some(clause) = new_tm.clauses.get_mut(cls) {
                    clause.positive_included_literals = vec![all_positive.into_iter().collect()];
                    clause.negative_included_literals = vec![all_negative.into_iter().collect()];
                }
            }
        }
    }
}

fn merge3(tms: &[TMClassifier], algo: &Algorithm) -> TMClassifier {
    let mut new_tm = tms.first().unwrap().clone();
    merge(&mut new_tm, tms, Some(algo));
    new_tm
}

fn combine(
    tms: Vec<(f64, TMClassifier)>,
    k: i64,
    x_test: &[TMInput],
    y_test: &[usize],
    algo: Option<&Algorithm>,
    batch: bool,
) -> (f64, TMClassifier) {
    let mut best: (f64, Option<TMClassifier>) = (0.0, None);
    let mut combinations = HashSet::new();

    let algo = algo.unwrap_or(&Algorithm::Merge);

    // Generate combinations of k classifiers
    // This is a placeholder for the actual combination generation logic
    for i in 0..tms.len() {
        combinations.insert(vec![i]);
    }

    println!(
        "Trying to find best combine accuracy among {} of {} combined models (algo: {:?}, batch size: {})...",
        combinations.len(),
        k,
        algo,
        tms.len()
    );

    let all_time = Instant::now();

    for (i, combination) in combinations.iter().enumerate() {
        let mut tm = tms[combination[0]].1.clone();

        let merging_time = Instant::now();
        for c in combinations.iter() {
            let a = c.iter().map(|t| tms[*t].1.clone()).collect::<Vec<_>>();
            merge(&mut tm, &a, Some(algo));
        }

        let merging_time = merging_time.elapsed();

        let testing_time = Instant::now();
        let predictions: Vec<Option<usize>> = predict_batch_2(&tm, x_test);

        let acc = accuracy(predictions, y_test);
        let testing_time = testing_time.elapsed();

        if acc >= best.0 {
            best = (acc, Some(tm));
        }

        println!(
            "#{}  Accuracy: {} = {:.2}%  Best: {:.2}%  Merging: {:.3}s  Testing: {:.3}s",
            i,
            combination
                .iter()
                .map(|&t| format!("{:.2}%", tms[t].0 * 100.0))
                .collect::<Vec<_>>()
                .join(" + "),
            acc * 100.0,
            best.0 * 100.0,
            merging_time.as_secs_f64(),
            testing_time.as_secs_f64()
        );
    }

    let elapsed = all_time.elapsed();
    println!(
        "Time elapsed: {:?}. Best {} combined models accuracy (algo: {:?}, batch size: {}): {:.2}%.\n",
        elapsed,
        k,
        algo,
        tms.len(),
        best.0 * 100.0
    );

    best.1
        .map_or_else(|| (0.0, TMClassifier::default()), |tm| (best.0, tm))
}
