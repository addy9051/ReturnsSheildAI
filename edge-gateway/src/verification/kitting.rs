//! High-Speed Packaging Tray Edge Energy and Bill of Materials (BOM) Auditor in Rust.

use std::collections::HashSet;
use std::path::Path;
use image::GrayImage;

use crate::models::{AccessoryStatus, BOMItem, KittedItemAudit, KittingAuditResult};

pub struct KittingAuditor {
    pub edge_energy_threshold: f64,
    pub entropy_threshold: f64,
}

impl Default for KittingAuditor {
    fn default() -> Self {
        Self {
            edge_energy_threshold: 12.0,
            entropy_threshold: 3.5,
        }
    }
}

impl KittingAuditor {
    pub fn new(edge_energy_threshold: f64, entropy_threshold: f64) -> Self {
        Self {
            edge_energy_threshold,
            entropy_threshold,
        }
    }

    /// Computes edge gradient energy and Shannon luminance entropy on a grayscale image slice.
    ///
    /// Why: Molded foam packaging exhibits uniform luminance with near-zero gradients. Populated
    /// slots (coiled USB cables, charging bricks, printed manuals) exhibit sharp edges and high entropy.
    pub fn compute_slot_metrics(img: &GrayImage) -> (f64, f64) {
        let (width, height) = img.dimensions();
        if width < 3 || height < 3 {
            return (0.0, 0.0);
        }

        // 1. Sobel Gradient Magnitude (3x3 kernel)
        let mut sum_magnitude = 0.0;
        let mut count = 0;

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let p00 = img.get_pixel(x - 1, y - 1).0[0] as f64;
                let p01 = img.get_pixel(x, y - 1).0[0] as f64;
                let p02 = img.get_pixel(x + 1, y - 1).0[0] as f64;

                let p10 = img.get_pixel(x - 1, y).0[0] as f64;
                let p12 = img.get_pixel(x + 1, y).0[0] as f64;

                let p20 = img.get_pixel(x - 1, y + 1).0[0] as f64;
                let p21 = img.get_pixel(x, y + 1).0[0] as f64;
                let p22 = img.get_pixel(x + 1, y + 1).0[0] as f64;

                // Sobel X: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
                let gx = (p02 + 2.0 * p12 + p22) - (p00 + 2.0 * p10 + p20);
                // Sobel Y: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
                let gy = (p20 + 2.0 * p21 + p22) - (p00 + 2.0 * p01 + p02);

                let mag = (gx * gx + gy * gy).sqrt();
                sum_magnitude += mag;
                count += 1;
            }
        }

        let edge_energy = if count > 0 { sum_magnitude / count as f64 } else { 0.0 };

        // 2. Shannon Luminance Entropy (32 bins)
        let mut hist = [0u32; 32];
        let total_pixels = (width * height) as f64;

        for pixel in img.pixels() {
            let bin = (pixel.0[0] / 8).min(31) as usize;
            hist[bin] += 1;
        }

        let mut entropy = 0.0;
        for &c in hist.iter() {
            if c > 0 {
                let p = c as f64 / total_pixels;
                entropy -= p * p.log2();
            }
        }

        (edge_energy, entropy)
    }

    /// Audits packaging tray presence against required SKU Bill of Materials.
    pub fn audit(
        &self,
        bom: &[BOMItem],
        detected_item_ids: Option<&[String]>,
        tray_crop_path: Option<&str>,
    ) -> KittingAuditResult {
        if bom.is_empty() {
            return KittingAuditResult {
                expected_items_count: 0,
                detected_items_count: 0,
                missing_items_count: 0,
                all_items_present: true,
                item_audits: Vec::new(),
                total_accessory_deduction: 0.0,
            };
        }

        let mut audits = Vec::new();
        let mut total_deduction = 0.0;

        // Mode A: Direct detected IDs provided (e.g. from optical classifier)
        if let Some(detected) = detected_item_ids {
            let detected_set: HashSet<&str> = detected.iter().map(|s| s.as_str()).collect();

            for item in bom {
                if detected_set.contains(item.item_id.as_str()) {
                    audits.push(KittedItemAudit {
                        bom_item_id: item.item_id.clone(),
                        name: item.name.clone(),
                        status: AccessoryStatus::PresentOem,
                        confidence: 0.98,
                        deduction_fee: 0.0,
                    });
                } else {
                    audits.push(KittedItemAudit {
                        bom_item_id: item.item_id.clone(),
                        name: item.name.clone(),
                        status: AccessoryStatus::Missing,
                        confidence: 0.95,
                        deduction_fee: item.deduction_fee,
                    });
                    total_deduction += item.deduction_fee;
                }
            }
        } else if let Some(path_str) = tray_crop_path {
            // Mode B: Fast visual image inspection
            let path = Path::new(path_str);
            if path.exists() {
                if let Ok(dyn_img) = image::open(path) {
                    let gray = dyn_img.to_luma8();
                    let (edge_energy, entropy) = Self::compute_slot_metrics(&gray);

                    let is_populated = edge_energy >= self.edge_energy_threshold
                        || entropy >= self.entropy_threshold;

                    let conf = (edge_energy / (self.edge_energy_threshold * 1.5))
                        .max(0.60)
                        .min(0.99);

                    for item in bom {
                        if is_populated {
                            audits.push(KittedItemAudit {
                                bom_item_id: item.item_id.clone(),
                                name: item.name.clone(),
                                status: AccessoryStatus::PresentOem,
                                confidence: (conf * 100.0).round() / 100.0,
                                deduction_fee: 0.0,
                            });
                        } else {
                            audits.push(KittedItemAudit {
                                bom_item_id: item.item_id.clone(),
                                name: item.name.clone(),
                                status: AccessoryStatus::Missing,
                                confidence: ((1.0 - conf * 0.3) * 100.0).round() / 100.0,
                                deduction_fee: item.deduction_fee,
                            });
                            total_deduction += item.deduction_fee;
                        }
                    }
                }
            }
        }

        // Default fallback if no image or detected list was supplied
        if audits.is_empty() {
            for item in bom {
                audits.push(KittedItemAudit {
                    bom_item_id: item.item_id.clone(),
                    name: item.name.clone(),
                    status: AccessoryStatus::Missing,
                    confidence: 0.50,
                    deduction_fee: item.deduction_fee,
                });
                total_deduction += item.deduction_fee;
            }
        }

        let detected_count = audits.iter().filter(|a| a.status == AccessoryStatus::PresentOem).count();
        let missing_count = bom.len() - detected_count;

        KittingAuditResult {
            expected_items_count: bom.len(),
            detected_items_count: detected_count,
            missing_items_count: missing_count,
            all_items_present: missing_count == 0,
            item_audits: audits,
            total_accessory_deduction: (total_deduction * 100.0).round() / 100.0,
        }
    }
}
