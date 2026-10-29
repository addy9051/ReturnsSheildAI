//! SIMD-ready Weighted Levenshtein Distance and Dual-Serial OCR Cross-Verifier in Rust.

use crate::models::{SerialMatchStatus, SerialVerificationResult};

pub struct SerialVerifier;

impl SerialVerifier {
    /// Normalizes serial strings by removing whitespace, hyphens, and converting to uppercase.
    pub fn normalize(s: &str) -> String {
        s.chars()
            .filter(|c| c.is_alphanumeric())
            .map(|c| c.to_ascii_uppercase())
            .collect()
    }

    /// Calculates substitution cost between two characters incorporating optical confusion penalties.
    #[inline(always)]
    pub fn substitution_cost(c1: char, c2: char) -> f64 {
        if c1 == c2 {
            return 0.0;
        }

        let pair = if c1 < c2 { (c1, c2) } else { (c2, c1) };

        match pair {
            // High-probability optical lookalikes (laser etch distortion)
            ('0', 'O') | ('1', 'I') | ('1', 'L') | ('I', 'L') => 0.15,
            // Secondary lookalikes
            ('5', 'S') | ('8', 'B') | ('2', 'Z') => 0.25,
            // General character mismatch
            _ => 1.0,
        }
    }

    /// Computes weighted edit distance using a dynamic programming 2-row buffer (cache-friendly O(N) space).
    pub fn weighted_levenshtein(s1: &str, s2: &str) -> f64 {
        let v1: Vec<char> = s1.chars().collect();
        let v2: Vec<char> = s2.chars().collect();
        let m = v1.len();
        let n = v2.len();

        if m == 0 {
            return n as f64;
        }
        if n == 0 {
            return m as f64;
        }

        let mut prev = vec![0.0; n + 1];
        let mut curr = vec![0.0; n + 1];

        for j in 0..=n {
            prev[j] = j as f64;
        }

        for i in 1..=m {
            curr[0] = i as f64;
            let c1 = v1[i - 1];

            for j in 1..=n {
                let c2 = v2[j - 1];
                let cost = Self::substitution_cost(c1, c2);

                let del = prev[j] + 1.0;
                let ins = curr[j - 1] + 1.0;
                let sub = prev[j - 1] + cost;

                curr[j] = del.min(ins).min(sub);
            }
            prev.copy_from_slice(&curr);
        }

        prev[n]
    }

    /// Cross-verifies scanned box shipping barcode against chassis laser serial engraving readout.
    pub fn verify(
        box_barcode_raw: Option<&str>,
        chassis_serial_raw: Option<&str>,
        _expected_regex: Option<&str>,
    ) -> SerialVerificationResult {
        let box_s = match box_barcode_raw {
            Some(s) if !s.trim().is_empty() => s.trim(),
            _ => {
                return SerialVerificationResult {
                    box_barcode: None,
                    chassis_serial_raw: chassis_serial_raw.map(|s| s.to_string()),
                    chassis_serial_normalized: None,
                    match_status: SerialMatchStatus::MissingLabel,
                    is_match: false,
                    similarity_ratio: 0.0,
                    ocr_confidence: 0.0,
                    notes: "Box barcode input is missing or empty.".to_string(),
                };
            }
        };

        let chassis_s = match chassis_serial_raw {
            Some(s) if !s.trim().is_empty() => s.trim(),
            _ => {
                return SerialVerificationResult {
                    box_barcode: Some(box_s.to_string()),
                    chassis_serial_raw: None,
                    chassis_serial_normalized: None,
                    match_status: SerialMatchStatus::OcrUnreadable,
                    is_match: false,
                    similarity_ratio: 0.0,
                    ocr_confidence: 0.0,
                    notes: "Chassis serial engraving unreadable or missing.".to_string(),
                };
            }
        };

        let norm_box = Self::normalize(box_s);
        let norm_chassis = Self::normalize(chassis_s);

        if norm_box == norm_chassis {
            return SerialVerificationResult {
                box_barcode: Some(box_s.to_string()),
                chassis_serial_raw: Some(chassis_s.to_string()),
                chassis_serial_normalized: Some(norm_chassis),
                match_status: SerialMatchStatus::ExactMatch,
                is_match: true,
                similarity_ratio: 1.0,
                ocr_confidence: 0.99,
                notes: format!("Exact character match verified: '{}'.", box_s),
            };
        }

        let max_len = norm_box.len().max(norm_chassis.len()) as f64;
        let weighted_dist = Self::weighted_levenshtein(&norm_box, &norm_chassis);
        let sim_ratio = (1.0 - (weighted_dist / max_len)).max(0.0);

        // Classification thresholds
        if weighted_dist <= 0.60 && sim_ratio >= 0.85 {
            SerialVerificationResult {
                box_barcode: Some(box_s.to_string()),
                chassis_serial_raw: Some(chassis_s.to_string()),
                chassis_serial_normalized: Some(norm_chassis),
                match_status: SerialMatchStatus::NormalizedMatch,
                is_match: true,
                similarity_ratio: (sim_ratio * 100.0).round() / 100.0,
                ocr_confidence: 0.92,
                notes: format!(
                    "Normalized OCR match confirmed (dist={:.2}, ratio={:.2}). Lookalike noise resolved.",
                    weighted_dist, sim_ratio
                ),
            }
        } else {
            SerialVerificationResult {
                box_barcode: Some(box_s.to_string()),
                chassis_serial_raw: Some(chassis_s.to_string()),
                chassis_serial_normalized: Some(norm_chassis),
                match_status: SerialMatchStatus::Mismatch,
                is_match: false,
                similarity_ratio: (sim_ratio * 100.0).round() / 100.0,
                ocr_confidence: 0.95,
                notes: format!(
                    "Genuine serial mismatch detected (dist={:.2}, ratio={:.2}). Stolen or swapped unit.",
                    weighted_dist, sim_ratio
                ),
            }
        }
    }
}
