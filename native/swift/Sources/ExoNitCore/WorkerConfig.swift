import Foundation

public enum WorkerFPPBackend: String, Codable, Sendable {
    case cpuSwift = "cpu-swift"
    case metal
}

public enum WorkerConfigError: Error, Equatable, CustomStringConvertible, Sendable {
    case invalidServerURL(String)
    case emptyToken
    case emptyWorkerName
    case invalidWorkerName(String)
    case invalidPositiveField(String)

    public var description: String {
        switch self {
        case let .invalidServerURL(value): return "Invalid HTTP(S) server URL: \(value)"
        case .emptyToken: return "Worker token must not be empty"
        case .emptyWorkerName: return "Worker name must not be empty"
        case let .invalidWorkerName(value): return "Invalid worker name: \(value)"
        case let .invalidPositiveField(name): return "\(name) must be positive"
        }
    }
}

public struct WorkerConfig: Codable, Equatable, Sendable {
    public var serverURL: URL
    public var token: String
    public var workerName: String
    public var cpuThreads: Int?
    public var gpuEnabled: Bool
    public var fppBackend: WorkerFPPBackend
    public var pollIntervalSeconds: Double
    public var heartbeatIntervalSeconds: Double
    public var fitsCacheDirectory: String
    public var guiPort: Int
    public var fppCPUCores: Int?
    public var logFile: String?
    public var logLevel: String
    public var exominerMode: String
    /// python `ffi_fallback_on_load_fail` (default false): when the FITS light
    /// curve cannot be loaded, fall back to the server's FFI cache. Only a JSON
    /// bool counts (anything else = off, with a warning).
    public var ffiFallbackOnLoadFail: Bool
    /// python `ffi_fetch_timeout` (seconds, default 30) of that FFI fetch.
    public var ffiFetchTimeout: Double
    public var startupCodeSync: Bool
    public var taskPools: [String]?
    /// Worker-level `gap_fill_detrend_only` (default false). A task-config key
    /// of the same name wins for primary tasks; FPP-backfill tasks have no
    /// config and use this value (python worker parity).
    public var gapFillDetrendOnly: Bool
    /// Worker-level `harmonic_epoch_class` (alias phase 2, default false). A
    /// bool in the task config wins (python worker precedence).
    public var harmonicEpochClass: Bool

    enum CodingKeys: String, CodingKey {
        case serverURL = "server_url"
        case token
        case workerName = "worker_name"
        case cpuThreads = "cpu_threads"
        case gpuEnabled = "gpu_enabled"
        case fppBackend = "fpp_backend"
        case pollIntervalSeconds = "poll_interval_seconds"
        case heartbeatIntervalSeconds = "heartbeat_interval_seconds"
        case fitsCacheDirectory = "fits_cache_dir"
        case guiPort = "gui_port"
        case fppCPUCores = "fpp_cpu_cores"
        case logFile = "log_file"
        case logLevel = "log_level"
        case exominerMode = "exominer_mode"
        case ffiFallbackOnLoadFail = "ffi_fallback_on_load_fail"
        case ffiFetchTimeout = "ffi_fetch_timeout"
        case startupCodeSync = "startup_code_sync"
        case taskPools = "task_pools"
        case gapFillDetrendOnly = "gap_fill_detrend_only"
        case harmonicEpochClass = "harmonic_epoch_class"
    }

    public init(
        serverURL: URL,
        token: String,
        workerName: String,
        cpuThreads: Int? = nil,
        gpuEnabled: Bool = false,
        fppBackend: WorkerFPPBackend = .cpuSwift,
        pollIntervalSeconds: Double = 5,
        heartbeatIntervalSeconds: Double = 30,
        fitsCacheDirectory: String = "fits_cache",
        guiPort: Int = 8402,
        fppCPUCores: Int? = nil,
        logFile: String? = nil,
        logLevel: String = "INFO",
        exominerMode: String = "legacy-tfrecord",
        ffiFallbackOnLoadFail: Bool = false,
        ffiFetchTimeout: Double = 30,
        startupCodeSync: Bool = false,
        taskPools: [String]? = nil,
        gapFillDetrendOnly: Bool = false,
        harmonicEpochClass: Bool = false
    ) {
        self.serverURL = serverURL
        self.token = token
        self.workerName = workerName
        self.cpuThreads = cpuThreads
        self.gpuEnabled = gpuEnabled
        self.fppBackend = fppBackend
        self.pollIntervalSeconds = pollIntervalSeconds
        self.heartbeatIntervalSeconds = heartbeatIntervalSeconds
        self.fitsCacheDirectory = fitsCacheDirectory
        self.guiPort = guiPort
        self.fppCPUCores = fppCPUCores
        self.logFile = logFile
        self.logLevel = logLevel
        self.exominerMode = exominerMode
        self.ffiFallbackOnLoadFail = ffiFallbackOnLoadFail
        self.ffiFetchTimeout = ffiFetchTimeout
        self.startupCodeSync = startupCodeSync
        self.taskPools = taskPools
        self.gapFillDetrendOnly = gapFillDetrendOnly
        self.harmonicEpochClass = harmonicEpochClass
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        serverURL = try c.decode(URL.self, forKey: .serverURL)
        token = try c.decode(String.self, forKey: .token)
        workerName = try c.decodeIfPresent(String.self, forKey: .workerName) ?? "worker"
        cpuThreads = try c.decodeIfPresent(Int.self, forKey: .cpuThreads)
        gpuEnabled = try c.decodeIfPresent(Bool.self, forKey: .gpuEnabled) ?? false
        fppBackend = try c.decodeIfPresent(WorkerFPPBackend.self, forKey: .fppBackend) ?? .cpuSwift
        pollIntervalSeconds = try c.decodeIfPresent(Double.self, forKey: .pollIntervalSeconds) ?? 5
        heartbeatIntervalSeconds = try c.decodeIfPresent(Double.self, forKey: .heartbeatIntervalSeconds) ?? 30
        fitsCacheDirectory = try c.decodeIfPresent(String.self, forKey: .fitsCacheDirectory) ?? "fits_cache"
        guiPort = try c.decodeIfPresent(Int.self, forKey: .guiPort) ?? 8402
        fppCPUCores = try c.decodeIfPresent(Int.self, forKey: .fppCPUCores)
        logFile = try c.decodeIfPresent(String.self, forKey: .logFile)
        logLevel = try c.decodeIfPresent(String.self, forKey: .logLevel) ?? "INFO"
        exominerMode = try c.decodeIfPresent(String.self, forKey: .exominerMode) ?? "legacy-tfrecord"
        if c.contains(.ffiFallbackOnLoadFail) {
            let raw = try c.decode(JSONValue.self, forKey: .ffiFallbackOnLoadFail)
            if case let .bool(value) = raw {
                ffiFallbackOnLoadFail = value
            } else {
                FileHandle.standardError.write(Data(
                    "WorkerConfig: ffi_fallback_on_load_fail=\(raw) is not a JSON bool; ignored (off)\n".utf8
                ))
                ffiFallbackOnLoadFail = false
            }
        } else {
            ffiFallbackOnLoadFail = false
        }
        ffiFetchTimeout = 30
        if c.contains(.ffiFetchTimeout) {
            let raw = try c.decode(JSONValue.self, forKey: .ffiFetchTimeout)
            switch raw {
            case let .number(v) where v.isFinite && v > 0: ffiFetchTimeout = v
            case let .integer(v) where v > 0: ffiFetchTimeout = Double(v)
            default:
                FileHandle.standardError.write(Data(
                    "WorkerConfig: ffi_fetch_timeout=\(raw) is not a positive number; using 30 s\n".utf8
                ))
            }
        }
        startupCodeSync = try c.decodeIfPresent(Bool.self, forKey: .startupCodeSync) ?? false
        taskPools = try c.decodeIfPresent([String].self, forKey: .taskPools)
        // Only a real JSON bool counts; any other value (string, number,
        // null) is ignored with a warning and reads as false -- same rule as
        // the Python and Rust workers, instead of rejecting the config.
        if c.contains(.gapFillDetrendOnly) {
            let raw = try c.decode(JSONValue.self, forKey: .gapFillDetrendOnly)
            if case let .bool(value) = raw {
                gapFillDetrendOnly = value
            } else {
                FileHandle.standardError.write(Data(
                    "WorkerConfig: gap_fill_detrend_only=\(raw) is not a JSON bool; ignored (off)\n".utf8
                ))
                gapFillDetrendOnly = false
            }
        } else {
            gapFillDetrendOnly = false
        }
        // Same rule: only a real JSON bool counts, anything else is off.
        if c.contains(.harmonicEpochClass) {
            let raw = try c.decode(JSONValue.self, forKey: .harmonicEpochClass)
            if case let .bool(value) = raw {
                harmonicEpochClass = value
            } else {
                FileHandle.standardError.write(Data(
                    "WorkerConfig: harmonic_epoch_class=\(raw) is not a JSON bool; ignored (off)\n".utf8
                ))
                harmonicEpochClass = false
            }
        } else {
            harmonicEpochClass = false
        }
    }

    public func validated() throws -> WorkerConfig {
        let scheme = serverURL.scheme?.lowercased()
        guard scheme == "http" || scheme == "https", serverURL.host != nil else {
            throw WorkerConfigError.invalidServerURL(serverURL.absoluteString)
        }
        guard !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw WorkerConfigError.emptyToken
        }
        guard !workerName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw WorkerConfigError.emptyWorkerName
        }
        guard WorkerNameValidator.isValid(workerName) else {
            throw WorkerConfigError.invalidWorkerName(workerName)
        }
        guard pollIntervalSeconds > 0 else { throw WorkerConfigError.invalidPositiveField("poll_interval_seconds") }
        guard heartbeatIntervalSeconds > 0 else { throw WorkerConfigError.invalidPositiveField("heartbeat_interval_seconds") }
        guard guiPort > 0 && guiPort <= 65_535 else { throw WorkerConfigError.invalidPositiveField("gui_port") }
        if let cpuThreads, cpuThreads <= 0 { throw WorkerConfigError.invalidPositiveField("cpu_threads") }
        if let fppCPUCores, fppCPUCores <= 0 { throw WorkerConfigError.invalidPositiveField("fpp_cpu_cores") }
        return self
    }

    public func applying(workerName: String?, serverURL: String?) throws -> WorkerConfig {
        var copy = self
        if let workerName { copy.workerName = workerName }
        if let serverURL {
            guard let parsed = URL(string: serverURL) else {
                throw WorkerConfigError.invalidServerURL(serverURL)
            }
            copy.serverURL = parsed
        }
        return try copy.validated()
    }
}
