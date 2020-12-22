# -*- coding:utf-8 -*-
"""
作者:wesley
日期:2020年12月22日
"""
from pyecharts.charts import Pie
from pyecharts import options as opts
x_data = ["小明", "小红", "张三", "李四", "王五"]
y_data = [335, 310, 234, 135, 548]

c = (
    Pie(init_opts=opts.InitOpts(width="1600px", height="1000px"))  # 图形的大小设置
        .add(
        series_name="访问来源",
        data_pair=[list(z) for z in zip(x_data, y_data)],
        radius=["15%", "50%"],  # 饼图内圈和外圈的大小比例
        center=["30%", "40%"],  # 饼图的位置：左边距和上边距
        label_opts=opts.LabelOpts(is_show=True),  # 显示数据和百分比
    )
        .set_global_opts(legend_opts=opts.LegendOpts(pos_left="left", orient="vertical"))  # 图例在左边和垂直显示
        .set_series_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
        ),
    )

)
c.render('环形.html')