use returnshield_edge::models::SerialMatchStatus;
use returnshield_edge::verification::SerialVerifier;

#[test]
fn test_serial_normalization() {
    assert_eq!(SerialVerifier::normalize("sn-sny-8842-109"), "SNSNY8842109");
    assert_eq!(SerialVerifier::normalize("SN DEV 5544"), "SNDEV5544");
}

#[test]
fn test_serial_exact_match() {
    let res = SerialVerifier::verify(Some("SN-AUD-100293"), Some("SN-AUD-100293"), None);
    assert!(res.is_match);
    assert_eq!(res.match_status, SerialMatchStatus::ExactMatch);
    assert_eq!(res.similarity_ratio, 1.0);
}

#[test]
fn test_serial_ocr_lookalike_confusion() {
    // Laser engraving misread: '0' misread as 'O', '1' misread as 'I'
    let box_code = "SN-AUD-880192";
    let chassis_ocr = "SN-AUD-88OI92";

    let res = SerialVerifier::verify(Some(box_code), Some(chassis_ocr), None);
    assert!(res.is_match);
    assert_eq!(res.match_status, SerialMatchStatus::NormalizedMatch);
    assert!(res.similarity_ratio >= 0.90);
}

#[test]
fn test_serial_genuine_mismatch() {
    let box_code = "SN-AUD-880192";
    let chassis_ocr = "SN-AUD-991044";

    let res = SerialVerifier::verify(Some(box_code), Some(chassis_ocr), None);
    assert!(!res.is_match);
    assert_eq!(res.match_status, SerialMatchStatus::Mismatch);
    assert!(res.similarity_ratio < 0.70);
}

