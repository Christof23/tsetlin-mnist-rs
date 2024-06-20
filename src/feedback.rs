#![allow(clippy::explicit_counter_loop)]

use ndarray::{Array2, Axis};
use rand::Rng;
use rand_xoshiro::Xoshiro256Plus;

use crate::{
    check_clause, TMInput, CLAUSE_SIZE, INCLUDE_LIMIT, L, NEG_R, R, STATES_MIN, STATES_NUM,
};

#[allow(clippy::too_many_arguments)]
pub fn feedback(
    x: &TMInput,
    clauses1: &mut Array2<u8>,
    clauses2: &mut Array2<u8>,
    literals1: &mut [Vec<u16>],
    literals2: &mut [Vec<u16>],
    update: f64,
    rng: &mut Xoshiro256Plus,
    j: &mut usize,
    row: &mut u16,
    count: &mut usize,
    literals_buffer: &mut [u16; 4704],
) {
    feedback1(
        clauses1,
        update,
        literals1,
        x,
        rng,
        literals_buffer,
        j,
        row,
        count,
    );
    feedback2(
        clauses2,
        rng,
        update,
        literals2,
        x,
        literals_buffer,
        j,
        row,
        count,
    );
}

#[allow(clippy::too_many_arguments)]
fn feedback1(
    clauses1: &mut Array2<u8>,
    update: f64,
    literals1: &mut [Vec<u16>],
    x: &TMInput,
    rng: &mut Xoshiro256Plus,
    literals_buffer: &mut [u16; CLAUSE_SIZE],
    j: &mut usize,
    row: &mut u16,
    count: &mut usize,
) {
    *j = 0;

    // use rand::SeedableRng;
    // use rayon::prelude::*;
    // clauses1
    //     .axis_iter_mut(Axis(1))
    //     .zip_eq(literals1)
    //     .enumerate()
    //     .par_bridge()
    //     .for_each(|(j, (mut c, literals))| {
    //         let mut rng: Xoshiro256Plus = Xoshiro256Plus::seed_from_u64(0);
    //         let mut literals_buffer = [0; CLAUSE_SIZE];
    //         if rng.gen_bool(update) {
    //             // let literals = literals1[j].as_slice();

    //             if check_clause(x, literals) {
    //                 if literals.len() <= L as usize {
    //                     for (c, x) in c.iter_mut().zip(x.x.iter()) {
    //                         if *x && *c < STATES_NUM {
    //                             *c += 1;
    //                         }
    //                     }
    //                 }

    //                 for (c, x) in c.iter_mut().zip(x.x.iter()) {
    //                     if rng.gen_bool(NEG_R) && !*x && *c < INCLUDE_LIMIT && *c > STATES_MIN {
    //                         *c -= 1;
    //                     }
    //                 }
    //             } else {
    //                 for value in &mut c {
    //                     if rng.gen_bool(NEG_R) && *value > STATES_MIN {
    //                         *value -= 1;
    //                     }
    //                 }
    //             }

    //             let mut count = 0;
    //             for (i, value) in c.iter().enumerate() {
    //                 if value >= &INCLUDE_LIMIT {
    //                     literals_buffer[count] = i as u16;
    //                     count += 1;
    //                 }
    //             }

    //             literals.clear();
    //             literals.extend_from_slice(&literals_buffer[0..count]);

    //             // literals1[j].clear();
    //             // literals1[j].extend_from_slice(&literals_buffer[0..count]);
    //         }
    //     });
    for mut c in clauses1.axis_iter_mut(Axis(1)) {
        if rng.gen_bool(update) {
            let literals = literals1[*j].as_slice();
            if check_clause(x, literals) {
                if literals.len() <= L as usize {
                    for (c, x) in c.iter_mut().zip(x.x.iter()) {
                        if *x && *c < STATES_NUM {
                            *c += 1;
                        }
                    }
                }

                for (c, x) in c.iter_mut().zip(x.x.iter()) {
                    if rng.gen_bool(NEG_R) && !*x && *c < INCLUDE_LIMIT && *c > STATES_MIN {
                        *c -= 1;
                    }
                }
            } else {
                for value in &mut c {
                    if rng.gen_bool(NEG_R) && *value > STATES_MIN {
                        *value -= 1;
                    }
                }
            }

            *count = 0;
            *row = 0;
            for value in c {
                if *value >= INCLUDE_LIMIT {
                    literals_buffer[*count] = *row;
                    *count += 1;
                }
                *row += 1;
            }

            literals1[*j].clear();
            literals1[*j].extend_from_slice(&literals_buffer[0..*count]);
        }
        *j += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn feedback2(
    clauses2: &mut Array2<u8>,
    rng: &mut Xoshiro256Plus,
    update: f64,
    literals2: &mut [Vec<u16>],
    x: &TMInput,
    literals_buffer: &mut [u16; CLAUSE_SIZE],
    j: &mut usize,
    row: &mut u16,
    count: &mut usize,
) {
    *j = 0;
    for mut c in clauses2.axis_iter_mut(Axis(1)) {
        let literals = literals2[*j].as_slice();
        if rng.gen_bool(update) && check_clause(x, literals) {
            for (c, x) in c.iter_mut().zip(x.x.iter()) {
                if rng.gen_bool(R) && !x && *c < INCLUDE_LIMIT {
                    *c += 1;
                }
            }

            *count = 0;
            *row = 0;
            for value in c {
                if *value >= INCLUDE_LIMIT {
                    literals_buffer[*count] = *row;
                    *count += 1;
                }
                *row += 1;
            }

            literals2[*j].clear();
            literals2[*j].extend_from_slice(&literals_buffer[0..*count]);
        }
        *j += 1;
    }
}
