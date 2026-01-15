import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import holidays


class OutlierCleaner(BaseEstimator, TransformerMixin):
    """
    Трансформер для очистки выбросов с поддержкой различных методов.

    Parameters
    ----------
    method : str, default='z'
        Метод определения выбросов: 'z' (z-score), 'iqr' (межквартильный размах), 
        'define' (пользовательские границы).
    left : float, default=3
        Левый множитель для метода 'z' или 'iqr', либо левая граница для 'define'.
    right : float, default=3
        Правый множитель для метода 'z' или 'iqr', либо правая граница для 'define'.
    kde_plt : bool, default=True
        Отображать ли KDE на графиках.
    action : str, default='remove'
        Действие с выбросами: 'remove' - удалить, 'clip' - обрезать до границ,
        'mask' - вернуть маску выбросов, 'none' - только вычисление границ.
    """

    def __init__(self, method='z', left=3, right=3, action='remove', mapping=None, kde_plt=True,  verbose=0):
        self.method = method
        self.left = left
        self.right = right
        self.action = action
        self.kde_plt = kde_plt
        self.mapping = mapping
        # self.y_params = y_params
        self.verbose = verbose
        self.cols_params = None

    def fit(self, X, y=None):
        """
        Вычисляет границы выбросов на основе переданных данных.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Входные данные.
        y : None
            Игнорируется, нужен для совместимости с API sklearn.

        Returns
        -------
        self : object
            Возвращает self.
        """
        if self.mapping is None:
            self.mapping = {col: {} for col in X.columns}

        self.cols_params = {}
        self.outliers_idxs = []
        for col, params in self.mapping.items():
            self.cols_params[col] = self.make_col_params(X[col], params)

        # if y is not None:
        #     data_ = pd.Series(y)
        #     self.y_params = self.make_col_params(data_, self.y_params)

        if self.verbose >= 2:
            self.plot_results_(X)

        return self

    def make_col_params(self, data: pd.Series, overparams):
        params = {}
        params['method'] = overparams.get('method', self.method)
        params['left'] = overparams.get('left', self.left)
        params['right'] = overparams.get('right', self.right)
        params['action'] = overparams.get('action', self.action)

        # Вычисление границ
        if params['method'] == 'z':
            mu = data.mean()
            sigma = data.std()
            params['left_bound'] = mu - params['left'] * sigma
            params['right_bound'] = mu + params['right'] * sigma

        elif params['method'] == 'iqr':
            quantile_1, quantile_3 = data.quantile(0.25), data.quantile(0.75)
            iqr = quantile_3 - quantile_1
            params['left_bound'] = quantile_1 - (iqr * params['left'])
            params['right_bound'] = quantile_3 + (iqr * params['right'])

        elif params['method'] == 'define':
            params['left_bound'] = params['left']
            params['right_bound'] = params['right']

        # Маска выбросов
        params['outlier_idxes'] = np.array(data[(data < params['left_bound']) | (data > params['right_bound'])].index)

        return params

    def transform(self, X):
        """
        Преобразует данные согласно выбранному действию.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            Входные данные.

        Returns
        -------
        X_transformed : array-like
            Преобразованные данные в зависимости от выбранного действия.
        """
        X = pd.DataFrame(X)
        for col, params in self.cols_params.items():
            # Применение выбранного действия
            if params['action'] == 'remove':
                data_ = X[col]
                # Возвращаем только не-выбросы
                X = X[(data_ >= params['left_bound']) & (data_ <= params['right_bound'])]

            elif params['action'] == 'clip':
                # Обрезаем выбросы до границ
                X[col] = X[col].clip(params['left_bound'], params['right_bound'])

            elif params['action'] == 'none':
                # Возвращаем оригинальные данные без изменений
                pass

            else:
                raise ValueError(f"Неизвестное действие: {self.action}")

        return X

    def plot_results_(self, X):
        """
        Визуализация результатов очистки выбросов.

        Parameters
        ----------
        data : array-like, optional
            Данные для визуализации. Если None, использует данные из fit.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            График с результатами.
        """
        limit_samples_ = 10000
        if X.shape[0] > limit_samples_:
            X = X.loc[np.random.choice(X.index, size=limit_samples_, replace=False)]

        for col, params in self.cols_params.items():
            X_series = X[col]

            outlier_mask = (X_series < params['left_bound']) | (X_series > params['right_bound'])
            cleaned = X_series[~outlier_mask]

            # Создание графика
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['До', 'После']
            )

            for i, d in enumerate([X_series, cleaned]):
                # Гистограмма
                fig.add_trace(
                    go.Histogram(
                        x=d,
                        xbins=dict(size=(d.max() - d.min()) / 50),
                        name='Гистограмма'
                    ),
                    row=1, col=i+1
                )

                # KDE
                if self.kde_plt and len(d) > 1:
                    try:
                        kde = gaussian_kde(d.dropna())
                        x_kde = np.linspace(d.min(), d.max(), 500)
                        y_kde_density = kde(x_kde)
                        y_kde_counts = (y_kde_density - np.min(y_kde_density)) / \
                            (np.max(y_kde_density) - np.min(y_kde_density)) * \
                            max(np.histogram(d, bins=50)[0])

                        fig.add_trace(
                            go.Scatter(
                                x=x_kde,
                                y=y_kde_counts,
                                mode='lines',
                                line=dict(color='red', width=2),
                                hoverinfo='text',
                                text=[f'{val:.4f}' for val in y_kde_density],
                                name='KDE',
                                hovertemplate='<b>KDE</b><br>x: %{x:.2f}<br>Density: %{text}<extra></extra>',
                                showlegend=False
                            ),
                            row=1, col=i+1
                        )
                    except:
                        pass

                fig.update_xaxes(title_text="Значение", row=1, col=i+1)
                fig.update_yaxes(title_text="Плотность", row=1, col=i+1)

                if i == 0:
                    # Добавляем вертикальные линии границ на первом графике
                    fig.add_vline(
                        x=params['left_bound'],
                        line_dash="dash",
                        line_color="green",
                        row=1, col=1
                    )
                    fig.add_vline(
                        x=params['right_bound'],
                        line_dash="dash",
                        line_color="green",
                        row=1, col=1
                    )

            # Обновление layout
            fig.update_layout(
                showlegend=False,
                width=1600,
                title=dict(
                    x=0.5,
                    text=col,
                    subtitle_text=(
                        f'Процент выбросов: {outlier_mask.sum() / X_series.shape[0] * 100:.2f}% | '
                        f'Границы: [{params['left_bound']:.3f}, {params['right_bound']:.3f}]'
                    )
                )
            )

            fig.show()


def extract_datetime_features(data: pd.DataFrame, get_holidays=True, holydays_params: dict = None):
    df = pd.DataFrame()
    data = pd.DataFrame(data)
    for col in data.columns:
        if not pd.api.types.is_datetime64_any_dtype(data):
            data[col] = pd.to_datetime(data[col])
            # raise TypeError('Признак', col, 'должен иметь тип данных datetime')

        # Извлекаем базовые признаки
        df[col+'_year'] = data[col].dt.year
        df[col+'_month'] = data[col].dt.month
        df[col+'_day'] = data[col].dt.day
        df[col+'_hour'] = data[col].dt.hour
        df[col+'_minute'] = data[col].dt.minute
        df[col+'_second'] = data[col].dt.second
        df[col+'_dayofweek'] = data[col].dt.dayofweek  # 0-6 (понедельник=0)
        df[col+'_dayofyear'] = data[col].dt.dayofyear
        df[col+'_weekofyear'] = data[col].dt.isocalendar().week
        df[col+'_quarter'] = data[col].dt.quarter
        df[col+'_is_weekend'] = data[col].dt.dayofweek.isin([5, 6]).astype(int)

        # Извлекаем праздники
        if get_holidays:
            def _days_since_last_holiday(dt, holiday_dict):
                """Дней с последнего праздника"""
                current_date = dt.date()
                holiday_dates = sorted([d for d in holiday_dict.keys() if d <= current_date])

                if not holiday_dates:
                    return np.nan

                last_holiday = holiday_dates[-1]
                return (current_date - last_holiday).days

            def _days_until_next_holiday(dt, holiday_dict):
                """Дней до следующего праздника"""
                current_date = dt.date()
                holiday_dates = sorted([d for d in holiday_dict.keys() if d >= current_date])

                if not holiday_dates:
                    return np.nan

                next_holiday = holiday_dates[0]
                return (next_holiday - current_date).days

            if holydays_params is None:
                holydays_params = {'country': 'US'}

            try:
                country_holidays = holidays.CountryHoliday(**holydays_params)
                df[col+'_holiday_name'] = data[col].dt.date.map(
                    lambda x: country_holidays.get(x, '')
                )
                df[col+'_is_holiday'] = df[col+'_holiday_name'].apply(lambda x: True if x else False)
                1
                # Типы праздников
                df[col+'_is_national_holiday'] = df[col+'_holiday_name'].apply(
                    lambda x: 1 if x and 'Day' in x or 'день' in x.lower() else 0
                )

                # Дни до/после праздника
                df[col+'_days_since_last_holiday'] = data[col].apply(
                    lambda x: _days_since_last_holiday(x, country_holidays)
                )
                df[col+'_days_until_next_holiday'] = data[col].apply(
                    lambda x: _days_until_next_holiday(x, country_holidays)
                )

            except:
                print(f"Не удалось загрузить праздники")
                df[col+'_is_holiday'] = 0
                df[col+'_holiday_name'] = ''

            # Сезонные и специальные периоды
            df[col+'_is_holiday_season'] = df[col+'_month'].isin([12, 1])  # Новогодние праздники
            df[col+'_is_summer_holidays'] = df[col+'_month'].isin([6, 7, 8])  # Летние каникулы

    return df
