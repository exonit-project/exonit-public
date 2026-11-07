//! Byte-exact port of CPython's `json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)`
//! for the `serde_json::Value` trees the worker posts (task 33 item 2.8: the 64 KiB `mc_detail` cap is measured
//! on that text, `worker._fpp_result_model_fields`).
//!
//! * floats use CPython's `float.__repr__` (shortest round-trip digits; exponent form when the decimal point
//!   position is <= -4 or > 16, two-digit minimum exponent, `.0` on integral values);
//! * strings use the default `ensure_ascii=True` escaping (`\uXXXX`, lowercase hex, surrogate pairs);
//! * object keys are sorted by code point (`String` byte order is code-point order for UTF-8).
//!
//! A serde `Value` cannot hold NaN or infinity, so `allow_nan=False` never fails here.

use serde_json::{Number, Value};

/// CPython `repr(float)`.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity".into() } else { "-Infinity".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    // Rust's `{:e}` without precision prints the shortest round-trip digits (same digits as CPython's repr).
    let s = format!("{x:e}");
    let (mant, exp) = s.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("exponent digits");
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1; // value = 0.<digits> * 10^decpt
    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if decpt <= -4 || decpt > 16 {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        out.push('e');
        out.push(if e < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", e.abs()));
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..(-decpt) {
            out.push('0');
        }
        out.push_str(&digits);
    } else {
        let d = decpt as usize;
        if d >= digits.len() {
            out.push_str(&digits);
            for _ in 0..(d - digits.len()) {
                out.push('0');
            }
            out.push_str(".0");
        } else {
            out.push_str(&digits[..d]);
            out.push('.');
            out.push_str(&digits[d..]);
        }
    }
    out
}

fn push_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_number(out: &mut String, n: &Number) {
    if let Some(u) = n.as_u64() {
        out.push_str(&u.to_string());
    } else if let Some(i) = n.as_i64() {
        out.push_str(&i.to_string());
    } else {
        out.push_str(&py_float_repr(n.as_f64().expect("a JSON number is an f64")));
    }
}

fn push_value(out: &mut String, v: &Value) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => push_number(out, n),
        Value::String(s) => push_string(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_value(out, x);
            }
            out.push(']');
        }
        Value::Object(o) => {
            let mut keys: Vec<&String> = o.keys().collect();
            keys.sort();
            out.push('{');
            for (i, k) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_string(out, k);
                out.push(':');
                push_value(out, &o[k]);
            }
            out.push('}');
        }
    }
}

/// `json.dumps(v, sort_keys=True, separators=(",", ":"))` (ensure_ascii=True).
pub fn canonical_dumps(v: &Value) -> String {
    let mut out = String::new();
    push_value(&mut out, v);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_cpython_on_the_format_boundaries() {
        for (x, want) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.5, "0.5"),
            (0.1, "0.1"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (123456.0, "123456.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e16, "1.2345e+16"),
            (1e22, "1e+22"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (0.052712591686247916, "0.052712591686247916"),
            (-2.5e-7, "-2.5e-07"),
            (9999999999999998.0, "9999999999999998.0"),
        ] {
            assert_eq!(py_float_repr(x), want, "{x:e}");
        }
    }

    #[test]
    fn strings_and_objects_follow_ensure_ascii_and_sort_keys() {
        let v: Value = serde_json::json!({"b": [1, 2.0, "x\"y"], "a": {"z": true, "\u{e9}": null}, "c": "\u{1f600}\n"});
        assert_eq!(
            canonical_dumps(&v),
            "{\"a\":{\"z\":true,\"\\u00e9\":null},\"b\":[1,2.0,\"x\\\"y\"],\"c\":\"\\ud83d\\ude00\\n\"}"
        );
    }
}
