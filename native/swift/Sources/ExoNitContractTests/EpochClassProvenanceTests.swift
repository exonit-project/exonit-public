import ExoNitCore
import Foundation

// Alias phase 2 provenance + task-config parsing, mirroring the python worker
// (`process_task`: task bool > worker config > off; result key
// `harmonic_epoch_class`; heartbeat flag_states key of the same name).

private func epochClassProcessor(workerFlag: Bool) throws -> PrimaryTaskProcessor {
    let directory = FileManager.default.temporaryDirectory
        .appendingPathComponent("exonit-hec-\(UUID().uuidString)", isDirectory: true)
    return try PrimaryTaskProcessor(
        api: WorkerAPIClient(
            baseURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", session: mockSession()
        ),
        cacheDirectory: directory, maximumConcurrency: 2, harmonicEpochClass: workerFlag
    )
}

func testEpochClassTaskPrecedenceTable() async throws {
    let taskCases: [(String?, Bool?)] = [  // (JSON literal or nil = absent, expected or nil = worker value)
        ("true", true), ("false", false), ("\"true\"", nil), ("\"false\"", nil),
        ("1", nil), ("0", nil), ("null", nil), (nil, nil),
    ]
    for worker in [false, true] {
        let processor = try epochClassProcessor(workerFlag: worker)
        try expect(processor.harmonicEpochClass(taskConfig: nil) == worker, "no task config -> worker")
        for (task, want) in [(JSONValue.bool(true), true), (.bool(false), false), (.string("false"), worker)] {
            let (options, effective) = processor.multiPlanetOptions(for: ["harmonic_epoch_class": task])
            try expect(options.harmonicEpochClass == want && effective == want, "options flag worker \(worker) task \(task)")
        }
        for (raw, expected) in taskCases {
            var config: [String: JSONValue] = [:]
            if let raw {
                config["harmonic_epoch_class"] = try JSONDecoder().decode(JSONValue.self, from: Data(raw.utf8))
            }
            try expect(
                processor.harmonicEpochClass(taskConfig: config) == (expected ?? worker),
                "worker \(worker) task \(raw ?? "absent")"
            )
        }
    }
}

func testEpochClassWorkerConfigAndHeartbeat() async throws {
    func decode(_ literal: String?) throws -> WorkerConfig {
        let extra = literal.map { #","harmonic_epoch_class":"# + $0 } ?? ""
        return try JSONDecoder().decode(
            WorkerConfig.self, from: Data((#"{"server_url":"http://localhost:8400","token":"t""# + extra + "}").utf8)
        )
    }
    for (literal, want) in [("true", true), ("false", false), ("\"true\"", false), ("1", false), ("null", false)] {
        let config = try decode(literal)
        try expect(config.harmonicEpochClass == want, "worker config \(literal)")
        // the heartbeat reports the EFFECTIVE value of the last task, never the config:
        // absent until a task has run, then whatever that task ran with
        try expect(WorkerRuntime.heartbeatFlagStates(config)["harmonic_epoch_class"] == nil, "no task yet \(literal)")
        try expect(
            WorkerRuntime.heartbeatFlagStates(config, effectiveHarmonicEpochClass: !want)["harmonic_epoch_class"] == .bool(!want),
            "effective value wins over config \(literal)"
        )
    }
    let absent = try decode(nil)
    try expect(!absent.harmonicEpochClass, "worker config default off")
    try expect(WorkerRuntime.heartbeatFlagStates(absent)["harmonic_epoch_class"] == nil, "heartbeat default: absent")
}

/// Heartbeat wiring: the runtime publishes what the processor's LAST task ran with
/// (task value > worker config > off), not the worker config; absent before any task.
func testEpochClassHeartbeatPublishesEffectiveValue() async throws {
    MockURLProtocol.handler = { request in
        (HTTPURLResponse(url: request.url!, statusCode: 500, httpVersion: nil, headerFields: nil)!, Data())
    }
    for configFlag in [false, true] {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("exonit-hb-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let config = try WorkerConfig(
            serverURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", workerName: "hb",
            fitsCacheDirectory: directory.path, taskPools: ["primary"], harmonicEpochClass: configFlag
        ).validated()
        let runtime = try WorkerRuntime(
            config: config, outboxURL: directory.appendingPathComponent("outbox.sqlite"),
            canaryAuthorized: true, session: mockSession()
        )
        try expect(runtime.currentHeartbeatFlagStates()["harmonic_epoch_class"] == nil, "absent before the first task (config \(configFlag))")
        for (taskValue, expected) in [(JSONValue.bool(!configFlag), !configFlag), (.bool(configFlag), configFlag)] {
            // a failing task (metadata 500) still records the flag it started with, as python does
            let result = try await runtime.primaryProcessor.process(
                WorkerTask(taskID: "t", ticID: "1", config: ["harmonic_epoch_class": taskValue]), workerID: "wid"
            )
            try expect(result.status == "failed", "task fails on the mock 500")
            try expect(
                runtime.currentHeartbeatFlagStates()["harmonic_epoch_class"] == .bool(expected),
                "effective \(expected) over config \(configFlag)"
            )
        }
    }
}

func testEpochClassResultWireKey() async throws {
    func json(_ result: PrimaryResultRequest) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: try JSONEncoder().encode(result)) as! [String: Any]
    }
    let on = try json(PrimaryResultRequest(taskID: "t", ticID: "1", harmonicEpochClass: true))
    try expect(on["harmonic_epoch_class"] as? Bool == true, "key present and true")
    let off = try json(PrimaryResultRequest(taskID: "t", ticID: "1", harmonicEpochClass: false))
    try expect(off["harmonic_epoch_class"] as? Bool == false, "key present and false")
    let absent = try json(PrimaryResultRequest(taskID: "t", ticID: "1", status: "failed", error: "load_failed"))
    try expect(absent["harmonic_epoch_class"] == nil, "failed result carries no key (python parity)")
}

private func epochClassFFIBody() -> Data {
    let dt = 10.0 / 1440.0
    var time: [Double] = [], flux: [Double] = []
    var state: UInt64 = 0x2545_F491_4F6C_DD1D
    for c in 0..<Int64(10.0 / dt) {
        state = state &* 6364136223846793005 &+ 1442695040888963407
        let u = Double(state >> 11) / Double(UInt64(1) << 53) - 0.5
        let t = 2000.0 + Double(c) * dt
        var f = 1.0 + 1.4e-3 * u
        if (t - 2000.37 + 23.0).truncatingRemainder(dividingBy: 2.3) / 2.3 < 0.02 { f -= 3e-3 }
        time.append(t); flux.append(f)
    }
    let object: [String: Any] = [
        "tic_id": "777", "sectors": [["sector": 14, "source": "tglc", "time": time, "flux": flux]],
    ]
    return try! JSONSerialization.data(withJSONObject: object)
}

func testEpochClassResultProvenanceEndToEnd() async throws {
    let ffi = epochClassFFIBody()
    MockURLProtocol.handler = { request in
        let path = request.url!.path
        func reply(_ code: Int, _ body: Data) -> (HTTPURLResponse, Data) {
            (HTTPURLResponse(url: request.url!, statusCode: code, httpVersion: nil, headerFields: nil)!, body)
        }
        if path.hasSuffix("/api/metadata/777") {
            return reply(200, Data(#"{"tic_id":"777","fits_files":[],"sectors":[14]}"#.utf8))
        }
        if path.hasSuffix("/api/ffi/777") { return reply(200, ffi) }
        return reply(404, Data(#"{"error":"not_found"}"#.utf8))
    }
    let small: [String: JSONValue] = [
        "n_periods": .integer(1500), "period_max": .number(8), "run_injection": .bool(false),
        "wotan_long_sweep": .bool(false), "max_planets": .integer(1),
    ]
    // (worker flag, task value or nil, effective)
    let cases: [(Bool, JSONValue?, Bool)] = [
        (false, nil, false), (true, nil, true), (false, .bool(true), true),
        (true, .bool(false), false), (true, .string("false"), true), (false, .null, false),
    ]
    for (worker, task, effective) in cases {
        var config = small
        if let task { config["harmonic_epoch_class"] = task }
        let result = try await epochClassProcessor(workerFlag: worker).process(
            WorkerTask(taskID: "t", ticID: "777", config: config), workerID: "wid"
        )
        try expect(result.status == "completed", "completed, got \(result.status) \(result.error ?? "")")
        try expect(
            result.harmonicEpochClass == effective,
            "worker \(worker) task \(String(describing: task)) -> \(String(describing: result.harmonicEpochClass))"
        )
    }
    // failed result (no light curve): no provenance key
    MockURLProtocol.handler = { request in
        let missing = HTTPURLResponse(url: request.url!, statusCode: 404, httpVersion: nil, headerFields: nil)!
        return (missing, Data(#"{"error":"not_found"}"#.utf8))
    }
    let failed = try await epochClassProcessor(workerFlag: true).process(
        WorkerTask(taskID: "t", ticID: "777"), workerID: "wid"
    )
    try expect(failed.status == "failed" && failed.harmonicEpochClass == nil, "failed result has no flag")
}

func testTaskConfigNullDoesNotCrash() async throws {
    let api = WorkerAPIClient(
        baseURL: URL(string: "http://127.0.0.1:18400")!, token: "test-token", session: mockSession()
    )
    for body in [
        #"{"task_id":"t1","tic_id":"123","config":null}"#,
        #"{"task_id":"t1","tic_id":123}"#,
    ] {
        MockURLProtocol.handler = { request in
            (HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!, Data(body.utf8))
        }
        guard case let .task(task) = try await api.pollTask(workerID: "wid") else {
            throw TestFailure(description: "expected a task for \(body)")
        }
        try expect(task.config.isEmpty && task.ticID == "123", "config null/absent -> empty config")
        let processor = try epochClassProcessor(workerFlag: false)
        try expect(!processor.harmonicEpochClass(taskConfig: task.config), "empty config -> OFF")
    }
}
