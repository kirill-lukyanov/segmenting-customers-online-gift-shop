import pandas as pd
from plotly.tools import mpl_to_plotly
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


def get_ts_info(ts: pd.Series, seas_decompose=True, order=True, order_a='5%', autocorrelation=True):
    # Вычисление порядка
    if order:
        max_order = 5
        order = 0
        diff = pd.Series(ts)
        while order <= max_order:
            adfuller_results = adfuller(diff)
            is_stationarity = adfuller_results[0] <= adfuller_results[4][order_a]
            print(f'Порядок - {order}, ряд', 'стационарен' if is_stationarity else 'нестационарен')
            if is_stationarity:
                break
            diff = diff.diff().dropna()
            order += 1

        print('Итоговый порядок d =', order)

    # Автокорреляция
    if autocorrelation:
        plot_acf(diff)
        plot_pacf(diff)

    # Декомпозиция
    if seas_decompose:
        fig = seasonal_decompose(ts).plot()
        fig = mpl_to_plotly(fig)
        fig.update_layout(width=1200)
        return fig
