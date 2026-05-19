# 这个仓库只放代码
# 每次工作前先同步
# 未经验证的工作不上传


 - [x] 处理图片 扩图加色彩x
 - [x] 心理学先验映射

## 清洗步骤  
- 准备三个文件夹 rawCsv slicedCsv processedCsv
- 然后将原始数据放入 rawCsv
- 使用 CLI_SliceCsv.py 得到切割后的数个Csv表格，用法在工具里边
- 使用 CLI_makeImgBigAndColorful.py 处理切割后的Csv文件
- 最终得到每1000行切片的Csv表格和img文件夹，Csv中只存图片路径
- 接着应用 CLI_emotion2VAVec2.py 替换 emotion 为 VA向量