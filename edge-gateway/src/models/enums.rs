//! Domain Enumerations for ReturnShield AI Edge Gateway.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FraudFlag {
    Clean,
    BaitAndSwitch,
    SerialMismatch,
    MissingAccessory,
    UnauthorizedAccessorySubstitution,
    PhysicalDamage,
    TamperedSerialLabel,
    Counterfeit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ConditionGrade {
    GradeA,
    GradeB,
    GradeC,
    FraudHold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DispositionAction {
    FullRefund,
    PartialRefund,
    RejectFraudHold,
    ManualSecondaryReview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SerialMatchStatus {
    ExactMatch,
    NormalizedMatch,
    Mismatch,
    MissingLabel,
    OcrUnreadable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum IdentityMatchStatus {
    VerifiedOem,
    SuspectedSwap,
    ConfirmedSwap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AccessoryStatus {
    PresentOem,
    Missing,
    SubstituteNonOem,
    Damaged,
}

