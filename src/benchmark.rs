fn benchmark(
    tm: &TMClassifier,
    x: &[TMInput],
    y: &[usize],
    loops: i64,
    batch: bool,
    deep_copy: bool,
    warmup: bool,
) {
    println!("CPU: {}", get_cpu_info()); // Assuming get_cpu_info() is defined elsewhere.

    println!("Preparing input data for benchmark...");
    let prepare_time = Instant::now();
    let mut rng = thread_rng();

    // Permutate in random order
    let mut perm: Vec<usize> = (0..(y.len() * loops as usize)).collect();
    perm.shuffle(&mut rng);

    // Multiply X and Y by loops times
    let mut x_perm: Vec<TMInput> = Vec::with_capacity(perm.len());
    let mut y_perm: Vec<usize> = Vec::with_capacity(perm.len());

    for &index in &perm {
        let loop_index = index % x.len();
        if deep_copy && !batch {
            x_perm.push(x[loop_index].clone()); // Assuming TMInput has a deep_copy method.
        } else {
            x_perm.push(x[loop_index].clone());
        }
        y_perm.push(y[loop_index % y.len()]);
    }

    if batch {
        x_perm = batches(&x_perm);
    }

    let prepare_duration = prepare_time.elapsed();
    println!(
        "Done. Elapsed {:.3} seconds.",
        prepare_duration.as_secs_f64()
    );

    if warmup {
        println!("Warm-up started in {} threads...", nthreads()); // Assuming nthreads() is defined elsewhere.
        let warmup_time = Instant::now();
        predict(tm, &x_perm); // Assuming predict() is defined elsewhere.
        let warmup_duration = warmup_time.elapsed();
        println!(
            "Done. Elapsed {:.3} seconds.",
            warmup_duration.as_secs_f64()
        );
    }

    let model_type = std::any::type_name::<TMClassifier>()
        .split('.')
        .last()
        .unwrap();
    if batch {
        println!(
            "Benchmark for {} model in batch mode (batch size = {}) started in {} threads...",
            model_type,
            usize::BITS,
            nthreads()
        );
    } else {
        println!(
            "Benchmark for {} model started in {} threads...",
            model_type,
            nthreads()
        );
    }

    let bench_time = Instant::now();
    let predicted = predict(tm, &x_perm);
    let bench_duration = bench_time.elapsed();
    let x_size = calculate_size(&x_perm); // Assuming calculate_size() is defined elsewhere.

    println!("Done.");
    println!(
        "{} predictions processed in {:.3} seconds.",
        predicted.len(),
        bench_duration.as_secs_f64()
    );
    println!(
        "Performance: {} predictions per second.",
        (predicted.len() as f64 / bench_duration.as_secs_f64()).floor()
    );
    println!(
        "Throughput: {:.3} GB/s.",
        (x_size as f64) / 1024f64.powi(3) / bench_duration.as_secs_f64()
    );
    println!(
        "Input data size: {:.3} GB.",
        (x_size as f64) / 1024f64.powi(3)
    );
    println!(
        "Parameters during training: {}.",
        tm.clauses_num * tm.clauses.len() as i64 * x[0].size()
    ); // Assuming TMInput has a size() method.
    println!(
        "Parameters after training and compilation: {}.",
        diff_count(tm).2
    ); // Assuming diff_count() is defined elsewhere.
    println!("Accuracy: {:.2}%", accuracy(&predicted, y) * 100f64); // Assuming accuracy() is defined elsewhere.
}
