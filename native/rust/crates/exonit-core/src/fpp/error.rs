//! Typed error kinds for FPP processing and task release policy.
//!
//! The release decision is carried by the error TYPE from the point where
//! the condition is detected (population validation, D-family index draw,
//! background LDC lookup, cross-check/alarm checks). No code path inspects
//! the message text to decide whether a task is released or failed; the
//! message is for logs only.
//!
//! Python raises two further `FPPUnavailable` kinds that have no Rust
//! counterpart because the data they guard is compiled into this crate:
//! `GeometryUnavailable` (`geometry_nodes_unavailable`, the smooth-geometry
//! node table is a missing/edited file in Python, `NODE_BITS` here) and the
//! `gl_tables_unavailable` host check (the Gauss-Legendre table is
//! `gl_tables.rs` here). Neither condition can occur at run time in Rust,
//! so no variant exists for them.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FppUnavailableReason {
    PopulationMissing,
    PopulationInvalid,
    PopulationCrosscheckUnverified,
    PopulationAlarmUnresolved,
    PopulationAlarm,
    PopulationLdcInvalid,
}

impl FppUnavailableReason {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::PopulationMissing => "population_missing",
            Self::PopulationInvalid => "population_invalid",
            Self::PopulationCrosscheckUnverified => "population_crosscheck_unverified",
            Self::PopulationAlarmUnresolved => "population_alarm_unresolved",
            Self::PopulationAlarm => "population_alarm",
            Self::PopulationLdcInvalid => "population_ldc_invalid",
        }
    }
}

impl std::fmt::Display for FppUnavailableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Why a candidate ended WITHOUT a score and without a release (`fpp = nfpp = -1`, `verdict = "failed"`):
/// Python's `calc_fpp_native.last_skip_reason`, posted as `skip_reason` (task 33 item 3.4). `None` on a
/// failure is Python's `None` (every other `return None` / generic exception arm).
///
/// Python also sets `ldc_table_missing` (`LDCTableMissing`) and `gl_tables_unavailable`; both guard data files
/// that are compiled into this crate (`stellar.rs` `include_str!`, `gl_tables.rs`), so neither can occur here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FppSkipReason {
    RawLcUnusable,
    GiantHost,
    StellarParamsMissing,
    LdcNoRow,
}

impl FppSkipReason {
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::RawLcUnusable => "raw_lc_unusable",
            Self::GiantHost => "giant_host",
            Self::StellarParamsMissing => "stellar_params_missing",
            Self::LdcNoRow => "ldc_no_row",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum FppError {
    Unavailable {
        reason: FppUnavailableReason,
        message: String,
    },
    Failed {
        message: String,
        skip_reason: Option<FppSkipReason>,
    },
}

impl FppError {
    pub fn unavailable(reason: FppUnavailableReason, message: impl Into<String>) -> Self {
        Self::Unavailable {
            reason,
            message: message.into(),
        }
    }

    pub fn failed(message: impl Into<String>) -> Self {
        Self::Failed {
            message: message.into(),
            skip_reason: None,
        }
    }

    /// A failure with Python's typed `skip_reason` (never derived from the message).
    pub fn skip(reason: FppSkipReason, message: impl Into<String>) -> Self {
        Self::Failed {
            message: message.into(),
            skip_reason: Some(reason),
        }
    }

    /// The `skip_reason` a `-1` post carries: `None` for a release and for an untyped failure.
    pub fn skip_reason(&self) -> Option<FppSkipReason> {
        match self {
            Self::Failed { skip_reason, .. } => *skip_reason,
            Self::Unavailable { .. } => None,
        }
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Unavailable { message, .. } => message,
            Self::Failed { message, .. } => message,
        }
    }

    pub fn unavailable_reason(&self) -> Option<FppUnavailableReason> {
        match self {
            Self::Unavailable { reason, .. } => Some(*reason),
            Self::Failed { .. } => None,
        }
    }
}

impl std::fmt::Display for FppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable { reason, message } => write!(f, "{}: {message}", reason.as_str()),
            Self::Failed { message, .. } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for FppError {}

impl From<crate::fpp::v2::population_v3::PopulationError> for FppError {
    fn from(err: crate::fpp::v2::population_v3::PopulationError) -> Self {
        Self::Unavailable {
            reason: err.reason,
            message: err.message,
        }
    }
}

/// Untyped (string) errors from the numerical code are ordinary failures,
/// exactly as Python treats every exception that is not `FPPUnavailable`
/// (`calc_fpp_native` logs it and returns `None`). A condition that must
/// release the task has to be constructed as `FppError::Unavailable` where
/// it is detected; it can never be recovered from a string.
impl From<String> for FppError {
    fn from(message: String) -> Self {
        Self::failed(message)
    }
}
