# 严格按照示例的导入和配置来（关键！）
import tushare as ts
import tushare.pro.client as client

# 示例里强调的：一定要加这行，否则报错
client.DataApi._DataApi__http_url = "http://tushare.xyz"

# 初始化pro接口（填你的token）
pro = ts.pro_api('c85afd3ab0ea551098bd6a0fbac1a19540b3e8ff279d07b1a340124d')

# 测试调取000001.SZ前复权日线数据（极简验证）
df = ts.pro_bar(
    ts_code='000001.SZ',  # 平安银行股票代码
    adj='qfq',            # 前复权（测试核心功能）
    start_date='20260301',# 最近日期，避免无数据
    end_date='20260322',
    asset='E',            # 资产类别：股票（必填）
    freq='D'              # 日线（必填）
)

# 打印结果，验证是否成功
print("复权行情数据（前5行）：")
print(df.head())

# 校验结果
if df is not None and not df.empty:
    print("\n✅ 接口调用成功！配置和复权功能都正常")
else:
    print("\n❌ 无数据返回，检查日期或token")