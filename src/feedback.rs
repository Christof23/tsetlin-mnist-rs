#![allow(clippy::explicit_counter_loop)]

use rand::Rng;
use rand_xoshiro::Xoshiro256Plus;

use crate::{
    check_clause, TMInput, CLAUSE_SIZE, INCLUDE_LIMIT, L, NEG_R, R, STATES_MIN, STATES_NUM,
};

#[allow(clippy::too_many_arguments)]
// #[inline]
pub fn feedback(
    x: &TMInput,
    clauses1: &mut [u8],
    clauses2: &mut [u8],
    literals1: &mut [Vec<u16>],
    literals2: &mut [Vec<u16>],
    update: f64,
    rng: &mut Xoshiro256Plus,
    count: &mut usize,
    literals_buffer: &mut [u16; CLAUSE_SIZE],
) {
    feedback1(x, clauses1, literals1, update, rng, literals_buffer, count);
    feedback2(x, clauses2, literals2, rng, update, literals_buffer, count);
}

#[allow(clippy::too_many_arguments)]
fn feedback1(
    x: &TMInput,
    clauses1: &mut [u8],
    literals1: &mut [Vec<u16>],
    update: f64,
    rng: &mut Xoshiro256Plus,
    literals_buffer: &mut [u16; CLAUSE_SIZE],
    count: &mut usize,
) {
    for (j, clauses) in clauses1.chunks_exact_mut(CLAUSE_SIZE).enumerate() {
        if rng.gen_bool(update) {
            if let Some(literals) = literals1.get(j) {
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
                for (row, clause) in clauses.iter().enumerate() {
                    if *clause >= INCLUDE_LIMIT {
                        if let Some(value) = literals_buffer.get_mut(*count) {
                            *value = row as u16;
                        }
                        *count += 1;
                    }
                }

                // Literals start of large then shrink with training
                if let Some(literals) = literals1.get_mut(j) {
                    literals.clear();
                    if let Some(buffer) = literals_buffer.get(0..*count) {
                        literals.extend_from_slice(buffer);
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]

fn feedback2(
    x: &TMInput,
    clauses2: &mut [u8],
    literals2: &mut [Vec<u16>],
    rng: &mut Xoshiro256Plus,
    update: f64,
    literals_buffer: &mut [u16; CLAUSE_SIZE],
    count: &mut usize,
) {
    for (j, clauses) in clauses2.chunks_exact_mut(CLAUSE_SIZE).enumerate() {
        if let Some(literals) = literals2.get(j) {
            if check_clause(x, literals) && rng.gen_bool(update) {
                for (c, x) in clauses.iter_mut().zip(x.x.iter()) {
                    if rng.gen_bool(R) && !x && *c < INCLUDE_LIMIT {
                        *c += 1;
                    }
                }

                *count = 0;
                for (row, c) in clauses.iter().enumerate() {
                    if *c >= INCLUDE_LIMIT {
                        if let Some(value) = literals_buffer.get_mut(*count) {
                            *value = row as u16;
                        }
                        *count += 1;
                    }
                }

                if let Some(literals) = literals2.get_mut(j) {
                    literals.clear();
                    if let Some(buffer) = literals_buffer.get(0..*count) {
                        literals.extend_from_slice(buffer);
                    }
                }
            }
        }
    }
}
