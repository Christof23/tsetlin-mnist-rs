#![allow(clippy::explicit_counter_loop)]

use rand::Rng;
use rand_xoshiro::Xoshiro256Plus;

use crate::{
    check_clause, TMInput, CLAUSE_SIZE, INCLUDE_LIMIT, L, NEG_R, R, STATES_MIN, STATES_NUM,
};

#[allow(clippy::too_many_arguments)]
pub fn feedback(
    x: &TMInput,
    clauses1: &mut [u8],
    clauses2: &mut [u8],
    literals1: &mut [Vec<u16>],
    literals2: &mut [Vec<u16>],
    update: f64,
    rng: &mut Xoshiro256Plus,
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
        row,
        count,
    );
}

#[allow(clippy::too_many_arguments)]
fn feedback1(
    clauses1: &mut [u8],
    update: f64,
    literals1: &mut [Vec<u16>],
    x: &TMInput,
    rng: &mut Xoshiro256Plus,
    literals_buffer: &mut [u16; CLAUSE_SIZE],
    row: &mut u16,
    count: &mut usize,
) {
    for (j, clauses) in clauses1.chunks_exact_mut(CLAUSE_SIZE).enumerate() {
        if rng.gen_bool(update) {
            let literals = literals1[j].as_slice();
            match check_clause(x, literals) {
                true => {
                    for (clause, &is_x_true) in clauses.iter_mut().zip(x.x.iter()) {
                        let is_within_literal_limit = literals.len() <= L as usize;
                        let can_increment_clause = is_x_true && *clause < STATES_NUM;
                        let can_decrement_clause =
                            !is_x_true && *clause < INCLUDE_LIMIT && *clause > STATES_MIN;

                        if is_within_literal_limit && can_increment_clause {
                            *clause += 1;
                        } else if rng.gen_bool(NEG_R) && can_decrement_clause {
                            *clause -= 1;
                        }
                    }
                }
                false => {
                    for clause in clauses.iter_mut() {
                        if *clause > STATES_MIN && rng.gen_bool(NEG_R) {
                            *clause -= 1;
                        }
                    }
                }
            }

            *count = 0;
            *row = 0;
            for clause in clauses {
                if *clause >= INCLUDE_LIMIT {
                    literals_buffer[*count] = *row;
                    *count += 1;
                }
                *row += 1;
            }

            // Literals start of large then shrink with training
            literals1[j].clear();
            literals1[j].extend_from_slice(&literals_buffer[0..*count]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn feedback2(
    clauses2: &mut [u8],
    rng: &mut Xoshiro256Plus,
    update: f64,
    literals2: &mut [Vec<u16>],
    x: &TMInput,
    literals_buffer: &mut [u16; CLAUSE_SIZE],
    row: &mut u16,
    count: &mut usize,
) {
    for (j, mut clauses) in clauses2.chunks_exact_mut(CLAUSE_SIZE).enumerate() {
        let literals: &[u16] = literals2[j].as_slice();

        if check_clause(x, literals) && rng.gen_bool(update) {
            for (c, x) in clauses.iter_mut().zip(x.x.iter()) {
                if rng.gen_bool(R) && !x && *c < INCLUDE_LIMIT {
                    *c += 1;
                }
            }

            *count = 0;
            *row = 0;
            for c in clauses {
                if *c >= INCLUDE_LIMIT {
                    literals_buffer[*count] = *row;
                    *count += 1;
                }
                *row += 1;
            }

            literals2[j].clear();
            literals2[j].extend_from_slice(&literals_buffer[0..*count]);
        }
    }
}
