import pyecharts.options as opts
from pyecharts.charts import Pie
from pyecharts.globals import ThemeType

# 嵌套饼图
# 内部饼图
inner_x_data = ["直达", "营销广告", "搜索引擎", "产品"]
inner_y_data = [335, 679, 548, 283]
inner_data_pair = [list(z) for z in zip(inner_x_data, inner_y_data)]
# [['直达', 335], ['营销广告', 679], ['搜索引擎', 1548], [‘产品’, 283]]

# 外部环形（嵌套）
outer_x_data = ["搜索引擎", "邮件营销", "直达", "营销广告", "联盟广告", "视频广告", "产品", "百度", "谷歌", "邮件营销", "联盟广告"]
outer_y_data = [335, 135, 147, 102, 220, 310, 234, 135, 648, 251]
outer_data_pair = [list(z) for z in zip(outer_x_data, outer_y_data)]

c = (
    # 初始化
    Pie(init_opts=opts.InitOpts(
        width="900px",  # 设置图形大小
        height="800px",
        theme=ThemeType.SHINE))  # 选择主题

        # 内部饼图
        .add(
        series_name="版本3.2.1",  # 图形名称
        center=["50%", "35%"],  # 饼图位置
        data_pair=inner_data_pair,  # 系列数据项，格式为 [(key1, value1), (key2, value2)]
        radius=["25%", "40%"],  # 饼图半径 数组的第一项是内半径，第二项是外半径
        label_opts=opts.LabelOpts(position='inner'),  # 标签设置在内部
    )

        # 外部嵌套环形图
        .add(
        series_name="版本3.2.9",  # 系列名称
        center=["50%", "35%"],  # 饼图位置
        radius=["40%", "60%"],  # 饼图半径 数组的第一项是内半径，第二项是外半径
        data_pair=outer_data_pair,  # 系列数据项，格式为 [(key1, value1), (key2, value2)]

        # 标签配置项
        label_opts=opts.LabelOpts(
            position="outside",
            formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
            background_color="#eee",
            border_color="#aaa",
            border_width=1,
            border_radius=4,
            rich={
                "a": {"color": "#999",
                      "lineHeight": 22,
                      "align": "center"},

                "abg": {
                    "backgroundColor": "#e3e3e3",
                    "width": "100%",
                    "align": "right",
                    "height": 22,
                    "borderRadius": [4, 4, 0, 0],
                },

                "hr": {
                    "borderColor": "#aaa",
                    "width": "100%",
                    "borderWidth": 0.5,
                    "height": 0,
                },

                "b": {"fontSize": 16, "lineHeight": 33},

                "per": {
                    "color": "#eee",
                    "backgroundColor": "#334455",
                    "padding": [2, 4],
                    "borderRadius": 2,
                },
            },
        ),
    )

        # 全局配置项
        .set_global_opts(
        xaxis_opts=opts.AxisOpts(is_show=False),  # 隐藏X轴刻度
        yaxis_opts=opts.AxisOpts(is_show=False),  # 隐藏Y轴刻度
        legend_opts=opts.LegendOpts(is_show=True),  # 隐藏图例
        title_opts=opts.TitleOpts(title=None),  # 隐藏标题
    )

        # 系统配置项
        .set_series_opts(
        tooltip_opts=opts.TooltipOpts(
            trigger="item",
            formatter="{a} <br/>{b}: {c} ({d}%)"
        ),
        label_opts=opts.LabelOpts(is_show=True)  # 隐藏每个触角标签
    )
)

c.render('pie.html')
