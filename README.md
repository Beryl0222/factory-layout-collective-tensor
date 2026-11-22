# 带围栏的异构张量转置交换

源端通道切片和目的端位置切片来自合成算子会话。物理页面字节值可离线检查。 数据为本地合成。

- `data/capture.json` 保存采集原始内容。records 是交换记录，tensors 字段分开描述名称、类型、形状和值。整数索引从零开始，未指定闭合的区间均为左闭右开。
- `data/capacity.json` 保存制造端的容量、规模或分片条件。大尺寸是接口需考虑的使用规模。工作区计入新分配的解包、转换、归约、版本暂存及交接缓冲。调用方已有只读输入和已消费的输出不计入工作区，额外物化副本计入。

## 交换约定

全局逻辑轴为 `N,C,H,W`，每元素为 little-endian uint16。每个逻辑盒按全局轴给出范围，`layout` 给出各盒的本地物理轴顺序；步幅按本地物理轴，以字节计。页字节偏移为基址加各局部索引乘相应步幅。页未被 data、fill 或 write_grants 显式涵盖的字节不可写。`fill` 字节窗及填充字也须在读取生效前建立。
目标写粒度为 `burst_align`，起点和长度均需对齐，每批单次最多 `link_max` 字节（有效载荷长度）。链路可重复任一片段。初始旧读者占用 `live_reads`，占用没有释放前，原代次涉及的字节仍须原样可读。`shared_grants` 允许所列目标在共用写范围内协商统一代次，未授权的共享写失败。`generation` 标识源图。

## 读取资料

使用 Python 3.11。首次创建本地环境时安装锁定依赖。只读查看显示原始字段和张量尺寸。

```sh
python -m pip install -r requirements.txt
python caseio.py data/capture.json
python -m unittest discover -s tests -v
```

现有检查核对采集格式和所附数据的结构。
