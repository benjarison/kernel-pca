use std::cmp::Ordering;
use num::Float;

// Get the indices of the elements of the provided vector sorted by descending value
//
// For example, 
// [0.5, 0.3, 0.8, 0.7, 0.4]
// should return
// [2, 3, 0, 4, 1]
pub fn sort_indices_descending<T: Float>(values: &[T]) -> Vec<usize> {
    let mut inds: Vec<usize> = (0..values.len()).collect();
    inds.sort_by(|&a, &b| {
        if values[a] < values[b] { 
            Ordering::Greater 
        } else { 
            Ordering::Less 
        }
    });
    return inds;
}
