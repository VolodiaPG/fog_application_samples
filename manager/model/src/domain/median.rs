use uom::si::f64::Time;

#[derive(Debug, Clone)]
pub struct Median {
    points: Vec<Time>,
}

/// Cumulative avg
impl Median {
    pub fn update(&mut self, latency: Time) { self.points.push(latency); }

    pub fn get_median(&self) -> Option<Time> { median(&self.points) }
}

impl Default for Median {
    fn default() -> Self { Median { points: Vec::new() } }
}

use std::cmp::Ordering;

fn partition(data: &[Time]) -> Option<(Vec<Time>, Time, Vec<Time>)> {
    match data.len() {
        0 => None,
        _ => {
            let (pivot_slice, tail) = data.split_at(1);
            let pivot = pivot_slice[0];
            let (left, right) =
                tail.iter().fold((vec![], vec![]), |mut splits, next| {
                    {
                        let (ref mut left, ref mut right) = &mut splits;
                        if next < &pivot {
                            left.push(*next);
                        } else {
                            right.push(*next);
                        }
                    }
                    splits
                });

            Some((left, pivot, right))
        }
    }
}

fn select(data: &[Time], k: usize) -> Option<Time> {
    let part = partition(data);

    match part {
        None => None,
        Some((left, pivot, right)) => {
            let pivot_idx = left.len();

            match pivot_idx.cmp(&k) {
                Ordering::Equal => Some(pivot),
                Ordering::Greater => select(&left, k),
                Ordering::Less => select(&right, k - (pivot_idx + 1)),
            }
        }
    }
}

fn median(data: &[Time]) -> Option<Time> {
    let size = data.len();

    match size {
        even if even % 2 == 0 => {
            let fst_med = select(data, (even / 2) - 1);
            let snd_med = select(data, even / 2);

            match (fst_med, snd_med) {
                (Some(fst), Some(snd)) => Some((fst + snd) as Time / 2.0),
                _ => None,
            }
        }
        odd => select(data, odd / 2).map(|x| x as Time),
    }
}
