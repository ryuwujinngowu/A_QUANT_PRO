-- 覆盖索引：优化 liquid_stats 因子的 60 日范围子查询
--
-- 问题：breakout SQL 中 WHERE trade_date BETWEEN x AND y GROUP BY ts_code
--       需要回表读取 close / amount 数据页（Extra: Using index condition; Using temporary）
--       实测耗时 2.14s
--
-- 修复：添加覆盖索引后，close / amount 直接从索引页读取，消除回表 I/O
--       预期 Extra 变为 Using index; Using temporary，耗时降至 ~0.3s
--
-- 影响评估：
--   - get_kline_day_range (ts_code IN + date range)：仍优先用 PRIMARY，无影响
--   - get_daily_kline_data (单日点查)：仍用 idx_trade_date，无影响
--   - 每日 ~5200 行 INSERT：维护额外索引代价极小（<1ms/行）
--
ALTER TABLE kline_day
  ADD INDEX idx_date_stock_cvr (trade_date, ts_code, close, amount);
