# A 股量化系统 v1.0 数据库核心文档



| 项目    | 内容                             |
| ----- | ------------------------------ |
| 版本    | v1.0                           |
| 核心操作  | 新增 7 张表，支撑 A 股基础数据 + 交易数据存储    |
| 适配数据库 | 阿里云 RDS MySQL 5.7+/8.0（InnoDB） |

## 一、新增表清单（表名 + 作用 + 核心字段类型）



| 表名                    | 核心作用               | 核心字段及类型                                                                                  |
| --------------------- | ------------------ | ---------------------------------------------------------------------------------------- |
| trade\_calendar       | 交易日校验（区分自然日 / 交易日） | exchange(CHAR(2)), trade\_date(DATE), is\_open(TINYINT)                                  |
| stock\_basic          | 股票基础信息存储（核心关联表）    | ts\_code(CHAR(8)), name(VARCHAR(32)), list\_date(DATE), industry(VARCHAR(32))            |
| stock\_board          | 板块 / 指数基础信息        | board\_code(CHAR(8)), board\_name(VARCHAR(32)), board\_type(VARCHAR(16))                 |
| stock\_board\_mapping | 股票 - 板块多对多映射       | id(BIGINT), ts\_code(CHAR(8)), board\_code(CHAR(8)), is\_valid(TINYINT)                  |
| stock\_adj\_factor    | 复权因子存储（计算复权价）      | ts\_code(CHAR(8)), trade\_date(DATE), adj\_factor(FLOAT), adj\_type(VARCHAR(8))          |
| kline\_day            | 日 K 线原始数据存储        | ts\_code(CHAR(8)), trade\_date(DATE), open(FLOAT), close(FLOAT), volume(BIGINT)          |
| kline\_1min           | 1 分钟 K 线原始数据存储     | ts\_code(CHAR(8)), trade\_time(DATETIME), trade\_date(DATE), open(FLOAT), volume(BIGINT) |

## 二、表间关联关系



1. **核心关联键**：`ts_code`（股票代码）是全局关联字段，关联`stock_basic`与`kline_day`/`kline_1min`/`stock_adj_factor`/`stock_board_mapping`；

2. **板块关联**：`stock_board_mapping`通过`board_code`关联`stock_board`，通过`ts_code`关联`stock_basic`，实现股票 - 板块多对多映射；

3. **日期关联**：`kline_day`/`kline_1min`/`stock_adj_factor`通过`trade_date`关联`trade_calendar`，校验交易日期合法性。