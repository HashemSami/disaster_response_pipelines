from plotly.graph_objects import Bar


def draw_bar(names, counts, title, x_title):
    graph = {
        "data": [Bar(x=names, y=counts)],
        "layout": {
            "title": title,
            "yaxis": {"title": "Count"},
            "xaxis": {"title": x_title},
            "height": 500,
            # 'paper_bgcolor': '#7f7f7f',
            # 'plot_bgcolor': '#c7c7c7'
        },
    }

    return graph


def draw_stacked_bar(genre_cat_counts, categories_names, title, x_title):
    traces = [
        {
            "x": list(genre_cat_counts.index),
            "y": genre_cat_counts[cat],
            "name": cat,
            "type": "bar",
        }
        for cat in categories_names
    ]

    graph = {
        "data": traces,
        "layout": {
            "barmode": "stack",
            "title": title,
            "yaxis": {"title": "Count"},
            "xaxis": {"title": x_title},
            "height": 1000,
            # 'paper_bgcolor': '#7f7f7f',
            # 'plot_bgcolor': '#c7c7c7'
        },
    }

    return graph


def draw_hor_bar(names, counts, title, x_title):
    graph = {
        "data": [Bar(x=counts, y=names, orientation="h")],
        "layout": {
            "title": title,
            "yaxis": {"title": ""},
            "xaxis": {"title": "Count"},
            "height": 1000,
            # 'paper_bgcolor': '#7f7f7f',
            # 'plot_bgcolor': '#c7c7c7'
        },
    }

    return graph
