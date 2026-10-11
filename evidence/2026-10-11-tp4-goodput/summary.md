# TP4 预登记场景汇总

状态：`passed`。探索、配对确认和串行确认分别列出。

主口径为每请求 TTFT≤1 s 且全部可观测单 token 间隔≤100 ms；四格同时保留。
Goodput 合计采用 qualified 总数 / 测量窗总时长；不平均分位数。

|阶段|方向/seed|角色|4格 qualified|token/s|pooled gap P99 ms|request worst-gap P99 ms|TTFT P99 ms|完整/eligible|
|---|---|---|---|---:|---:|---:|---:|---|
|pilot|exploration/101|native|0/0/1/1 of 38|112.650|219.835|234.508|17787.094|True/True|
|pilot|exploration/101|plugin|38/38/38/38 of 38|136.867|60.380|78.116|953.017|True/True|
|pilot|exploration/101|native|1/1/3/4 of 38|127.050|209.143|230.921|4995.293|True/True|
|pilot|exploration/101|plugin|38/38/38/38 of 38|137.167|60.866|80.549|925.019|True/True|
|pilot|exploration/101|native|5/5/16/16 of 38|137.417|193.553|468.034|565.520|True/True|
|pilot|exploration/101|plugin|38/38/38/38 of 38|136.900|57.446|81.085|921.729|True/True|
|heldout_paired|first/1001|native|22/22/33/33 of 71|113.592|22.529|462.924|629.263|True/True|
|heldout_paired|first/1001|plugin|66/67/70/71 of 71|114.142|60.589|154.470|930.397|True/True|
|heldout_paired|first/1002|native|19/19/30/30 of 57|89.608|20.658|470.159|720.487|True/True|
|heldout_paired|first/1002|plugin|57/57/57/57 of 57|88.458|56.746|81.918|939.163|True/True|
|heldout_paired|first/1003|native|17/17/29/29 of 61|107.600|22.269|465.931|595.238|True/True|
|heldout_paired|first/1003|plugin|60/60/61/61 of 61|109.708|56.708|95.836|897.666|True/True|
|heldout_paired|crossover/1001|native|23/23/37/37 of 71|113.642|22.379|460.856|637.380|True/True|
|heldout_paired|crossover/1001|plugin|67/67/71/71 of 71|114.175|58.361|129.104|819.559|True/True|
|heldout_paired|crossover/1002|native|19/19/30/30 of 57|89.608|20.247|454.692|733.187|True/True|
|heldout_paired|crossover/1002|plugin|57/57/57/57 of 57|88.492|54.261|76.741|905.591|True/True|
|heldout_paired|crossover/1003|native|17/17/26/26 of 61|107.500|21.060|453.347|591.369|True/True|
|heldout_paired|crossover/1003|plugin|61/61/61/61 of 61|109.542|55.033|91.148|873.598|True/True|
|serial|serial/1001|native|22/22/33/33 of 71|113.517|22.920|464.373|622.051|True/True|
|serial|serial/1001|plugin|68/68/71/71 of 71|114.142|57.759|104.371|810.355|True/True|

四格顺序为 100 ms/1 s、100 ms/2 s、200 ms/1 s、200 ms/2 s。

每方向三个 seed 共用同一服务启动，不能视为三次独立启动。两个角色并行共享 host；串行结果单列。
这是固定到达率、固定应用场景下的确认，不证明最大可持续容量或全场景优势。
pooled gap P99 与每请求 worst-gap P99 描述不同现象，两者及 TTFT 的代价必须同时报告。

`heldout_paired`：{"throughput_ratio_plugin_over_native": 1.0047730137023034, "joint_slos": [{"gap_limit_ms": 100, "ttft_limit_ms": 1000, "native_qualified": 117, "plugin_qualified": 368, "goodput_ratio_plugin_over_native": 3.145299145299145, "zero_native_goodput": false}, {"gap_limit_ms": 100, "ttft_limit_ms": 2000, "native_qualified": 117, "plugin_qualified": 369, "goodput_ratio_plugin_over_native": 3.1538461538461533, "zero_native_goodput": false}, {"gap_limit_ms": 200, "ttft_limit_ms": 1000, "native_qualified": 185, "plugin_qualified": 377, "goodput_ratio_plugin_over_native": 2.0378378378378383, "zero_native_goodput": false}, {"gap_limit_ms": 200, "ttft_limit_ms": 2000, "native_qualified": 185, "plugin_qualified": 378, "goodput_ratio_plugin_over_native": 2.0432432432432437, "zero_native_goodput": false}]}
`serial`：{"throughput_ratio_plugin_over_native": 1.005505799442079, "joint_slos": [{"gap_limit_ms": 100, "ttft_limit_ms": 1000, "native_qualified": 22, "plugin_qualified": 68, "goodput_ratio_plugin_over_native": 3.090909090909091, "zero_native_goodput": false}, {"gap_limit_ms": 100, "ttft_limit_ms": 2000, "native_qualified": 22, "plugin_qualified": 68, "goodput_ratio_plugin_over_native": 3.090909090909091, "zero_native_goodput": false}, {"gap_limit_ms": 200, "ttft_limit_ms": 1000, "native_qualified": 33, "plugin_qualified": 71, "goodput_ratio_plugin_over_native": 2.1515151515151514, "zero_native_goodput": false}, {"gap_limit_ms": 200, "ttft_limit_ms": 2000, "native_qualified": 33, "plugin_qualified": 71, "goodput_ratio_plugin_over_native": 2.1515151515151514, "zero_native_goodput": false}]}
