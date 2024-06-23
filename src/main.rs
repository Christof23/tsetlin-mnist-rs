use std::ops::Index;
use std::str::FromStr;
use std::time::Instant;
use std::vec;

use rand::prelude::SliceRandom;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

mod feedback;

use crate::feedback::feedback;

const EPOCHS: i64 = 1;
const NUM_CLAUSES: i64 = 128;
const T: i64 = 8;
const R: f64 = 0.89;
const NEG_R: f64 = 1.0 - R;
const L: i8 = 16;
const BEST_TMS_SIZE: i64 = 500;
const STATES_MIN: u8 = 0;
const STATES_NUM: u8 = 255;
const INCLUDE_LIMIT: u8 = 128;
const CLAUSE_SIZE: usize = 4704; // 28*28*3*2

const SAMPLES: Option<usize> = Some(1000);

fn main() {
    let train_path = "mnist/mnist_train.csv";
    let test_path = "mnist/mnist_test.csv";

    let (y_train, x_train) = read_mnist(train_path, SAMPLES);
    let (y_test, x_test) = read_mnist(test_path, None);

    println!(
        "Train samples: {}\nTest samples: {}",
        y_train.len(),
        y_test.len()
    );

    // Training the TM model
    let mut tm = TMClassifier::new(NUM_CLAUSES, T, R, L, STATES_NUM, INCLUDE_LIMIT);

    let (_best_accuracy, _best_tms) = train_model(
        &mut tm,
        &x_train,
        &y_train,
        &x_test,
        &y_test,
        EPOCHS,
        true,
        1,
        BEST_TMS_SIZE,
    );
}

fn read_mnist(path: &str, samples: Option<usize>) -> (Vec<usize>, Vec<TMInput>) {
    let (mut y, mut x): (Vec<usize>, Vec<TMInput>) = std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| {
            let (y, x) = line.split_once(',').unwrap();
            (usize::from_str(y).unwrap(), TMInput::new(x, None))
        })
        .unzip();

    if samples.is_some() {
        x = x.get(..samples.unwrap()).unwrap().to_vec();
        y = y.get(..samples.unwrap()).unwrap().to_vec();
    }

    (y, x)
}

#[derive(Debug, Clone, Default)]
struct TATeam {
    positive_clauses: Vec<u8>,
    negative_clauses: Vec<u8>,
    positive_included_literals: Vec<Vec<u16>>,
    negative_included_literals: Vec<Vec<u16>>,
}

impl TATeam {
    fn new(clauses_num: i64, include_limit: u8) -> Self {
        let positive_clauses: Vec<u8> =
            vec![include_limit - 1; CLAUSE_SIZE * (clauses_num / 2) as usize];
        let negative_clauses: Vec<u8> =
            vec![include_limit - 1; CLAUSE_SIZE * (clauses_num / 2) as usize];

        let positive_included_literals = vec![vec![0; CLAUSE_SIZE]; (clauses_num / 2) as usize];
        let negative_included_literals = vec![vec![0; CLAUSE_SIZE]; (clauses_num / 2) as usize];

        assert_eq!(negative_included_literals.len(), 64);
        assert_eq!(negative_clauses.len(), 4704 * 64);

        Self {
            positive_clauses,
            negative_clauses,
            positive_included_literals,
            negative_included_literals,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct TMClassifier {
    clauses_num: i64,
    t: i64,
    r: f64,
    l: i8,
    include_limit: u8,
    state_max: u8,
    clauses: Vec<TATeam>, // clauses: HashMap<usize, TATeam>
}

impl TMClassifier {
    fn new(clauses_num: i64, t: i64, r: f64, l: i8, states_num: u8, include_limit: u8) -> Self {
        Self {
            clauses_num,
            t,
            r,
            l,
            include_limit,
            state_max: states_num,
            clauses: Vec::with_capacity(10),
        }
    }
}

#[derive(Clone, Debug)]
struct TMInput {
    x: Vec<bool>,
}

impl TMInput {
    // The `new` function takes a vector of booleans and an optional `negate` parameter.
    // If `negate` is true, it appends the negated vector to the original one.
    fn new(s: &str, negate: Option<bool>) -> Self {
        let mut x: Vec<bool> = s
            .split(',')
            .map(|a| a.parse::<f64>().expect("Failed to parse float"))
            .flat_map(|num| vec![num > 0.0, num > 84.0, num > 169.0]) // 0.0, 0.33 and 0.66 in 0-255 range.
            .collect::<Vec<_>>();

        if negate.unwrap_or(true) {
            let mut negated_x = x.iter().map(|&value| !value).collect::<Vec<bool>>();
            x.append(&mut negated_x);
        }

        TMInput { x }
    }

    // Mimics Julia's `getindex` method
    fn get_index(&self, i: usize) -> Option<bool> {
        self.x.get(i).copied() // Returns an Option type for safety
    }
}

// More idomatic indexing

impl Index<usize> for TMInput {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        &self.x[index]
    }
}

fn initialize(tm: &mut TMClassifier, x: &[TMInput]) {
    let clause_size = x.first().unwrap().x.len() as i64;
    assert_eq!(clause_size, 4704);
    for _ in 0..10 {
        let ta_team = TATeam::new(tm.clauses_num, tm.include_limit);
        tm.clauses.push(ta_team);
    }
}

// for each literal feature, if all values for the sample are true,
// then the clause is true

fn check_clause(x: &TMInput, literals: &[u16]) -> bool {
    literals
        .iter()
        .all(|&literal| x.get_index(literal as usize).unwrap_or(false))
}

fn vote(ta: &TATeam, x: &TMInput) -> i64 {
    ta.positive_included_literals
        .iter()
        .zip(ta.negative_included_literals.iter())
        .map(|(pos_lit, neg_lit)| check_clause(x, pos_lit) as i64 - check_clause(x, neg_lit) as i64)
        .sum()
}

fn get_v(t: i64, ta_team: &TATeam, x: &TMInput) -> i64 {
    let v: i64 = vote(ta_team, x);
    v.clamp(-t, t)
}

fn get_update_value(t: i64, ta_team: &TATeam, x: &TMInput, positive: bool) -> f64 {
    let v: i64 = get_v(t, ta_team, x);
    let update: f64 = if positive {
        (t - v) as f64
    } else {
        (t + v) as f64
    } / (t * 2) as f64;

    update
}

#[allow(clippy::too_many_arguments)]
fn feedback_positive(
    tm: &mut TMClassifier,
    x: &TMInput,
    y: usize,
    positive: bool,
    rng: &mut Xoshiro256Plus,
    row: &mut u16,
    count: &mut usize,
    literals_buffer: &mut [u16; 4704],
) {
    let ta_team: &mut TATeam = tm.clauses.get_mut(y).unwrap();

    let update: f64 = get_update_value(tm.t, ta_team, x, positive);

    let clauses1 = &mut ta_team.positive_clauses;
    let clauses2 = &mut ta_team.negative_clauses;
    let literals1: &mut Vec<Vec<u16>> = &mut ta_team.positive_included_literals;
    let literals2: &mut Vec<Vec<u16>> = &mut ta_team.negative_included_literals;

    feedback(
        x,
        clauses1,
        clauses2,
        literals1,
        literals2,
        update,
        rng,
        row,
        count,
        literals_buffer,
    );
}

#[allow(clippy::too_many_arguments)]
fn feedback_negative(
    tm: &mut TMClassifier,
    x: &TMInput,
    y: usize,
    positive: bool,
    rng: &mut Xoshiro256Plus,
    row: &mut u16,
    count: &mut usize,
    literals_buffer: &mut [u16; 4704],
) {
    let ta_team: &mut TATeam = tm.clauses.get_mut(y).expect("OOB");

    let update: f64 = get_update_value(tm.t, ta_team, x, positive);

    let clauses1 = &mut ta_team.negative_clauses;
    let clauses2 = &mut ta_team.positive_clauses;
    let literals1: &mut Vec<Vec<u16>> = &mut ta_team.negative_included_literals;
    let literals2: &mut Vec<Vec<u16>> = &mut ta_team.positive_included_literals;

    feedback(
        x,
        clauses1,
        clauses2,
        literals1,
        literals2,
        update,
        rng,
        row,
        count,
        literals_buffer,
    );
}

fn predict_batch_2(tm: &TMClassifier, x: &[TMInput]) -> Vec<Option<usize>> {
    let mut best_vote: Vec<i64> = vec![i64::MIN; x.len()];
    let mut best_cls: Vec<Option<usize>> = vec![None; x.len()];

    for (cls, ta_team) in tm.clauses.iter().enumerate() {
        for (i, x) in x.iter().enumerate() {
            let v: i64 = get_v(tm.t, ta_team, x);
            if v > best_vote[i] {
                best_vote[i] = v;
                best_cls[i] = Some(cls);
            }
        }
    }

    best_cls
}

fn accuracy(predicted: Vec<Option<usize>>, y: &[usize]) -> f64 {
    let correct: usize = predicted
        .iter()
        .zip(y.iter())
        .filter(|(p, &y)| p.is_some() && p.unwrap() == y)
        .count();

    correct as f64 / y.len() as f64
}

fn train(
    tm: &mut TMClassifier,
    x: &TMInput,
    y: usize,
    classes: &mut [usize; 10],
    shuffle: bool,
    rng: &mut Xoshiro256Plus,
) {
    assert_eq!(x.x.len(), 4704);

    if shuffle {
        classes.shuffle(rng);
    }

    let mut row = 0u16;
    let mut count = 0usize;
    let mut literals_buffer: [u16; 4704] = [0u16; CLAUSE_SIZE];

    for cls in classes {
        if *cls != y {
            feedback_positive(
                tm,
                x,
                y,
                true,
                rng,
                &mut row,
                &mut count,
                &mut literals_buffer,
            );
            feedback_negative(
                tm,
                x,
                *cls,
                false,
                rng,
                &mut row,
                &mut count,
                &mut literals_buffer,
            );
        }
    }

    // println!("{:#?}", tm.clauses[0].positive_included_literals.len());
}

fn train_batch(tm: &mut TMClassifier, x: &[TMInput], y: &[usize], shuffle: bool) {
    // let mut rng = thread_rng();
    // let mut rng: XorShiftRng = XorShiftRng::seed_from_u64(123456789);

    // let mut rng: SmallRng = rand::rngs::SmallRng::from_entropy();
    let mut rng: Xoshiro256Plus = Xoshiro256Plus::seed_from_u64(42);

    // If not initialized yet
    if tm.clauses.is_empty() {
        initialize(tm, x); // Assuming initialize is defined elsewhere
    }

    let mut data: Vec<_> = x.iter().zip(y.iter()).collect();

    if shuffle {
        data.shuffle(&mut rng);
    }

    let mut classes: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    for (input, &output) in data {
        // Here we call the train function for each input-output pair

        // let now = Instant::now();
        train(tm, input, output, &mut classes, shuffle, &mut rng);
        // let elapsed = now.elapsed();
        // println!("Sample: {i} elapsed: {:#?}", elapsed);
    }
}

#[allow(clippy::too_many_arguments)]
fn train_model(
    tm: &mut TMClassifier,
    x_train: &[TMInput],
    y_train: &[usize],
    x_test: &[TMInput],
    y_test: &[usize],
    epochs: i64,
    shuffle: bool,
    verbose: u8,
    best_tms_size: i64,
) -> (f64, TMClassifier) {
    assert!((1..=2000).contains(&best_tms_size));

    // let num_cpus: u32 = sys_info::cpu_num().unwrap();

    if verbose > 0 {
        // println!("\nRunning in {} threads.", num_cpus);
        println!(
            "Accuracy over {} epochs (Clauses: {}, T: {}, R: {}, L: {}, states_num: {}, include_limit: {}):\n",
            epochs,
            tm.clauses_num,
            tm.t,
            tm.r,
            tm.l,
            tm.state_max + 1,
            tm.include_limit
        );
    }

    let mut best_tm: (f64, Option<TMClassifier>) = (0.0, None);

    let all_time = Instant::now();

    for i in 0..epochs {
        let now = Instant::now();
        train_batch(tm, x_train, y_train, shuffle);
        let training_time = now.elapsed().as_millis();

        let now = Instant::now();
        // 69ms

        let y_pred = predict_batch_2(tm, x_test);
        let testing_time = now.elapsed().as_millis();
        let acc = accuracy(y_pred, y_test);

        if acc >= best_tm.0 {
            best_tm = (acc, Some(tm.clone()));
        }

        // best_tms.push((
        //     acc,
        //     if best_tms_compile {
        //         let tm_classifier_compiled = compile(tm, verbose - 1);
        //         AbstractTMClassifier::TMClassifierCompiled(tm_classifier_compiled)
        //     } else {
        //         AbstractTMClassifier::TMClassifier(tm.clone())
        //     },
        // ));
        // best_tms.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        // best_tms.truncate(best_tms_size as usize);

        if verbose > 0 {
            println!(
                "#{}  Accuracy: {:.2}%  Best: {:.2}%  Training: {}ms  Testing: {}ms",
                i,
                acc * 100.0,
                best_tm.0 * 100.0,
                training_time,
                testing_time
            );
        }
    }

    if verbose > 0 {
        println!(
            "\nDone. {} epochs (Clauses: {}, T: {}, R: {}, L: {}, states_num: {}, include_limit: {}).",
            epochs,
            tm.clauses_num,
            tm.t,
            tm.r,
            tm.l,
            tm.state_max + 1,
            tm.include_limit
        );
        let elapsed = all_time.elapsed();
        println!(
            "Time elapsed: {:?}. Best accuracy was: {:.2}%.\n",
            elapsed,
            best_tm.0 * 100.0
        );
    }
    (best_tm.0, best_tm.1.unwrap())
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_bit_set() {
        let a: i64 = 0b1000_0010_0010_0010_0000_0010_1000_0000_0000_0010_0000_0010;
        let b: i64 = 0b1000_0010_0010_0010_0000_0010_1000_0010_0010_0010_0100_0011;

        let q = (a >> 1) & 1;
        println!("{:#?}", q);
        // let b = 0b0000_0010;
        let now = Instant::now();
        let c = a | b;
        let elapsed = now.elapsed();
        println!("{:#?}", elapsed.as_nanos());
        println!("{:#010b}", c);
        // println!("{:#?}", a);

        let a = [
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        ];
        let b = [
            1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        ];

        let now = Instant::now();
        let c = a
            .iter()
            .zip(b.iter())
            .map(|(a, b)| a | b)
            .collect::<Vec<_>>();
        let elapsed = now.elapsed();
        println!("{:#?}", elapsed.as_nanos());
        // println!("{:#?}", c);
        // use bitvec::prelude::*;
        let mut a = bit_vec::BitVec::from_elem(1000, true);
        // for i in 0..1000 {
        //     a.set(i, i % 2 == 0);
        // }

        let now = Instant::now();
        let rs = a.and(&a.clone());
        let elapsed = now.elapsed();
        println!("{:#?}", elapsed.as_nanos());
        println!("{:#?}", rs);
        // println!("{:#018b}", c);
        // println!("{:#?}", rs);
        // a.set(1, true);
        // let mut b = bitarr![u8, Lsb0; 1];
        // let mut b = bit_vec::BitVec::from_elem(10, false);
        // a.set(0, true);

        // let b = bitarr![i8, Lsb0; 0; 5];
        // let c = a & b;
        // let c = a.and(&b);
        // println!("{:#?}", a);
        // use bit_vec::BitVec;

        // let max_prime = 10;

        // // Store the primes as a BitVec
        // let primes = {
        //     // Assume all numbers are prime to begin, and then we
        //     // cross off non-primes progressively
        //     let mut bv = BitVec::from_elem(max_prime, true);

        //     // Neither 0 nor 1 are prime
        //     bv.set(0, false);
        //     bv.set(1, false);

        //     for i in 2..1 + (max_prime as f64).sqrt() as usize {
        //         // if i is a prime
        //         if bv[i] {
        //             // Mark all multiples of i as non-prime (any multiples below i * i
        //             // will have been marked as non-prime previously)
        //             for j in i.. {
        //                 if i * j >= max_prime {
        //                     break;
        //                 }
        //                 bv.set(i * j, false)
        //             }
        //         }
        //     }
        //     bv
        // };

        // println!("primes: {:#?}", primes);

        // let mut a = BitVec::from_elem(1, true);
        // a.set(0, true);
        // a.set(1, true);

        // let mut b = BitVec::from_elem(1, true);
        // b.set(0, true);
        // b.set(1, true);
        // println!("{:#?} {:#?}", a, b);

        // let c = a |= b;
        // for c in a {

        // }
        // let c = a.or(&b);
        // println!("{:#?} {:#?} {:#?}", a, b, c);
        // assert!(c);

        // assert_eq!(a, bv);

        // let mut a = BitVec::from_elem(1, true);
        // let b = BitVec::from_elem(1, true);

        // let res = a.or(&b);
        // println!("{:#?}", res);
        // println!("res: {:#?}", res);
    }
}
