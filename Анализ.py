#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Данный ноутбук принимает на вход таблицу формата csv, содержащую следующие колонки:
# Год: int
# Муниципалитет: str
# Пол: str
# Возрастная группа: str
# Причина смерти: str
# Стандартный вес группы: str, должен быть указан в процентах (например '4,4')
# Число умерших: int
# Численность населения: int, имеется в виду численность населения возрастной группы


# In[2]:


import pandas as pd
import numpy as np
from scipy import stats
import math
pd.set_option('display.max_colwidth', None)


# In[3]:


# Загрузка исходной таблицы.
df = pd.read_csv("Тестовое _ Данные - Лист1.csv")


# In[4]:


# Определение уровня значимости, который будет использоваться для расчётов.
alpha = 0.05
# Расчёт уровня доверия.
CL = int((1-alpha)*100)


# # Содержание таблицы

# In[5]:


print('df', '\n')
print(df.head(50).to_string(), '\n', '\n')


# In[6]:


# Диапазоны значений колонок
print('Диапазоны значений колонок:', '\n')

for column in df.columns:
    print(column, df[column].unique().tolist(), '\n', '\n')


# # Вычисление коэффициентов

# ## Приведение входных данных к удобному виду

# In[7]:


# Приведение колонки 'Стандартный вес группы' исходной таблицы к виду 0.01 (т.е. доли, а не проценты).
df['Стандартный вес группы'] = df['Стандартный вес группы'].str.replace(',', '.').astype(float) / 100


# ## Функции

# In[8]:


def get_slice(
    df: pd.DataFrame,
    district: str,
    year: int,
    target: str = None):

    """
    Делает срез по исходным данным по требуемым в задании параметрам.
    df: исходный датафрейм
    district: муниципалитет
    year: год
    target: пол, причина смерти или None
    Возвращает датафрейм формата входных данных.
    """
    if year is None:
        if target is None:
            df_slice = (df[(df['Муниципалитет']==district)]
                .groupby(['Возрастная группа', 'Стандартный вес группы', 'Пол'], as_index=False)
                .agg({'Число умерших': 'sum', 'Численность населения': 'mean'}))
            df_slice = (df_slice.groupby(['Возрастная группа', 'Стандартный вес группы'], as_index=False)
                .agg({'Число умерших': 'sum', 'Численность населения': 'sum'}))
        
        elif target in df['Причина смерти'].unique():
            df_slice = (df[(df['Муниципалитет']==district)&(df['Причина смерти']==target)]
                .groupby(['Возрастная группа', 'Стандартный вес группы', 'Пол'], as_index=False)
                .agg({'Число умерших': 'sum', 'Численность населения': 'mean'}))
            df_slice = (df_slice.groupby(['Возрастная группа', 'Стандартный вес группы'], as_index=False)
                .agg({'Число умерших': 'sum', 'Численность населения': 'sum'}))
        
        elif target in df['Пол'].unique():
            df_slice = (df[(df['Муниципалитет']==district)&(df['Пол']==target)]
                .groupby(['Возрастная группа', 'Стандартный вес группы'], as_index=False)
                .agg({'Число умерших': 'sum', 'Численность населения': 'mean'}))
        
        else:
            raise ValueError("Неверное значение target, выполнение прервано")
    
    else:
        if target is None:
            df_slice = (df[(df['Муниципалитет']==district)&(df['Год']==year)]
                .groupby(['Возрастная группа', 'Стандартный вес группы', 'Пол'], as_index=False)
                .agg({'Число умерших': 'sum', 'Численность населения': 'first'}))
            df_slice = (df_slice.groupby(['Возрастная группа', 'Стандартный вес группы'], as_index=False)
                .agg({'Число умерших': 'sum', 'Численность населения': 'sum'}))
        
        elif target in df['Причина смерти'].unique():
            df_slice = (df[(df['Муниципалитет']==district)&(df['Год']==year)&(df['Причина смерти']==target)]
                .groupby(['Возрастная группа', 'Стандартный вес группы', 'Пол'], as_index=False)
                .agg({'Число умерших': 'sum', 'Численность населения': 'first'}))
            df_slice = (df_slice.groupby(['Возрастная группа', 'Стандартный вес группы'], as_index=False)
                .agg({'Число умерших': 'sum', 'Численность населения': 'sum'}))
        
        elif target in df['Пол'].unique():
            df_slice = (df[(df['Муниципалитет']==district)&(df['Год']==year)&(df['Пол']==target)]
                .groupby(['Возрастная группа', 'Стандартный вес группы'], as_index=False)
                .agg({'Число умерших': 'sum', 'Численность населения': 'first'}))
        
        else:
            raise ValueError("Неверное значение target, выполнение прервано")
        
    return df_slice


# In[9]:


def N_D_SCR_CI_calc(
    df: pd.DataFrame,
    alpha: float = 0.05):

    """
    Рассчитывает cуммарную численность населения, суммарное число умерших,
    стандартизованный коэффициент смертности, нижнюю и верхнюю границы доверительного интервала для СКС,
    статистическую значимость отличия СКС от нуля по исходной таблице.
    Границы доверительного интервала рассчитываются методом нормального приближения распределения Пуассона для числа смертей больше 25
    и методом Добсона для числа смертей от 10 до 25. При числе смертей меньше 10 в качестве значений границ возвращается None.
    df: датафрейм формата исходных данных
    alpha: уровень значимости
    Возвращает суммарную численность населения,суммарное число умерших,
    стандартизованный коэффициент смертности, нижнюю и верхнюю границы доверительного интервала для СКС,
    статистическую значимость отличия СКС от нуля.
    """

    # Создание колонки 'Коэффициент смертности'.
    df['Коэффициент смертности'] = df['Число умерших'] / df['Численность населения']
    # Вычисление суммарной численности населения по срезу.
    N = df['Численность населения'].sum()
    # Вычисление суммарного числа умерших по срезу.
    D = df['Число умерших'].sum()
    # Вычисление стандартизованного коэффициента смертности на 100000 человек для среза.
    SCR = (df['Коэффициент смертности'] * df['Стандартный вес группы']).sum() * 100000
    if D > 9:
        if D < 25:
            # Вычисление стандартной ошибки стандартизованного коэффициента смертности на 100000 человек для среза (метод Добсона).
            LCL = SCR + ((((df['Стандартный вес группы'].pow(2) * df['Коэффициент смертности'] / df['Численность населения']).sum() /
                           df['Число умерших'].sum()) ** 0.5)
                         * (stats.chi2.ppf(alpha/2, 2 * df['Число умерших'].sum())/2 - df['Число умерших'].sum()) * 100000)
            UCL = SCR + ((((df['Стандартный вес группы'].pow(2) * df['Коэффициент смертности'] / df['Численность населения']).sum() /
                           df['Число умерших'].sum()) ** 0.5)
                         * (stats.chi2.ppf(1 - alpha/2, 2 * df['Число умерших'].sum() + 2)/2 - df['Число умерших'].sum()) * 100000)
        else:
            # Вычисление стандартной ошибки стандартизованного коэффициента смертности на 100000 человек для среза
            # (пуассоновская аппроксимация нормального распределения).
            LCL = SCR - (((df['Стандартный вес группы'].pow(2) * df['Коэффициент смертности'] / df['Численность населения']).sum() ** 0.5)
                         * stats.norm.ppf(1 - alpha/2) * 100000)
            UCL = SCR + (((df['Стандартный вес группы'].pow(2) * df['Коэффициент смертности'] / df['Численность населения']).sum() ** 0.5)
                         * stats.norm.ppf(1 - alpha/2) * 100000)
        # Определение статистической значимости отличия СКС от нуля.
        if LCL > 0:
            return N.item(), D.item(), SCR.item(), LCL.item(), UCL.item(), 'Да'
        else:
            return N.item(), D.item(), SCR.item(), LCL.item(), UCL.item(), 'Нет'
    else:
        return N.item(), D.item(), SCR.item(), None, None, 'Нет'


# In[10]:


def N_D_SCR_CI_by_slice(
    df: pd.DataFrame,
    district: str,
    year: int = None,
    target: str = None,
    alpha: int = 0.05):

    """
    Расчитывает суммарную численность населения, суммарное число умерших,
    стандартизованный коэффициент смертности и границы доверительного интервала для СКС
    посредством функций get_slice и N_SCR_CI_calc.
    df: исходный датафрейм
    district: муниципалитет
    year: год
    target: пол, причина смерти или None
    Возвращает суммарную численность населения,суммарное число умерших,
    стандартизованный коэффициент смертности, нижнюю и верхнюю границы доверительного интервала для СКС.
    """

    # Создание таблицы среза через вызов функции get_slice.
    df_slice = get_slice(df=df, district=district, year=year, target=target)
    return N_D_SCR_CI_calc(df_slice, alpha)


# ## Вычисление

# In[11]:


# Создание списка годов.
all_years_list = df['Год'].unique().tolist()
# Создание списка причин смерти.
all_causes_list = df['Причина смерти'].unique().tolist()
# Создание списка муниципалитетов.
all_districts_list = df['Муниципалитет'].unique().tolist()
# Создание списка полов.
all_genders_list = df['Пол'].unique().tolist()


# ### Вычисление по годам

# In[12]:


# Создание набора данных из всех комбинаций всех уникальных значений колонок 'Муниципалитет' и 'Год'.
targets_district_year_list = [[x, y] for x in df['Муниципалитет'].unique() for y in df['Год'].unique()]

# Создание набора данных из всех комбинаций всех уникальных значений колонок 'Муниципалитет', 'Год' и 'Пол'.
targets_district_year_gender_list = [[x, y, z] for x in df['Муниципалитет'].unique() \
                                    for y in df['Год'].unique() for z in df['Пол'].unique()]

# Создание набора данных из всех комбинаций всех уникальных значений колонок 'Муниципалитет', 'Год' и 'Причина смерти'.
targets_district_year_death_cause_list = [[x, y, z] for x in df['Муниципалитет'].unique() \
                                          for y in df['Год'].unique() for z in df['Причина смерти'].unique()]


# In[13]:


# Названия колонок будующей таблицы с суммарным разрезом и разрезами по гендеру и причинам смерти.

columns_list = ['Муниципалитет', 'Год', 'Пол', 'Причина смерти', 'Численность населения', 'Число умерших', 'СКС',
                'Нижняя граница ДИ ({}%)'.format(CL), 'Верхняя граница ДИ ({}%)'.format(CL), 'Значимость СКС (p < {})'.format(alpha)]


# In[14]:


# Суммарный разрез (объединяет данные по гендеру и причинам смерти сохраняя муниципалитет и год).
# Расчёт значений суммарной численности населения, суммарного числа умерших, СКС и границ ДИ для СКС,
# и включение этих данных в содержимое листа result_district_year_list
result_district_year_list = [row + ['Все', 'Все'] + list(N_D_SCR_CI_by_slice(df, row[0], row[1])) \
                             for row in targets_district_year_list]
# Создание датафрейма
df_result_district_year = pd.DataFrame(result_district_year_list, columns=columns_list)

# Разрез по гендеру (объединяет данные по гендеру сохраняя муниципалитет, год и причины смерти)
# Расчёт значений суммарной численности населения, суммарного числа умерших, СКС и границ ДИ для СКС,
# и включение этих данных в содержимое листа result_district_year_gender_list
result_district_year_gender_list = [row + ['Все'] + list(N_D_SCR_CI_by_slice(df, row[0], row[1], row[2])) \
                             for row in targets_district_year_gender_list]
# Создание датафрейма
df_result_district_year_gender = pd.DataFrame(result_district_year_gender_list, columns=columns_list)

# Разрез по причинам смерти (объединяет данные по причинам смерти сохраняя муниципалитет, год и гендер)
# Расчёт значений суммарной численности населения, суммарного числа умерших, СКС и границ ДИ для СКС,
# и включение этих данных в содержимое листа result_district_year_death_cause_list
result_district_year_death_cause_list = [[row[0], row[1], 'Все', row[2]] + list(N_D_SCR_CI_by_slice(df, row[0], row[1], row[2])) \
                             for row in targets_district_year_death_cause_list]
# Создание датафрейма
df_result_district_year_death_cause = pd.DataFrame(result_district_year_death_cause_list, columns=columns_list)

# Создание общего датафрейма
df_result = pd.concat([df_result_district_year, df_result_district_year_gender, df_result_district_year_death_cause], ignore_index=True)


# In[15]:


print(df_result, '\n')
print(df_result.head(50).to_string(), '\n', '\n')


# In[16]:


# Подготовка таблицы для выгрузки: только нужные колонки.
df_export = df_result[['Муниципалитет', 'Год', 'Пол', 'Причина смерти', 'СКС']].copy()
# Сохранение в CSV.
df_export.to_csv('СКС_результаты.csv', index=False, encoding='utf-8-sig')

# Округление до целых.
df_export['СКС'] = df_export['СКС'].round(0).astype(int)
# Сохранение в CSV.
df_export.to_csv('СКС_результаты_округление_до_целых.csv', index=False, encoding='utf-8-sig')


# # Интерпретация

# ## Функции

# ### Статика

# In[17]:


def comparison(
    SCR_1: float,
    LCL_1: float,
    UCL_1: float,
    SCR_2: float,
    LCL_2: float,
    UCL_2: float):

    """
    Рассчитывает значимость различия между двумя значениями стандартизованного коэффициента смертности и показывает какое из них больше.
    Если >, то первое, если <, то второе.
    SCR_1: первый стандартизованный коэффициент смертности
    LCL_1: нижняя граница ДИ для первого СКС
    UCL_1: верхняя граница ДИ для первого СКС
    SCR_2: второй стандартизованный коэффициент смертности
    LCL_2: нижняя граница ДИ для второго СКС
    UCL_2: верхняя граница ДИ для второго СКС
    Возвращает статистически значимое отличие SCR_1 от SCR_2 в формате '>', '=', '<' или None
    в случае незначимости данных по хотябы одной группе.
    """

    # Проверка на значимость данных элементов сравниваемой пары.
    if LCL_1 is None or np.isnan(LCL_1) or LCL_2 is None or np.isnan(LCL_2):
        return None

    # Проверка на равенство нулю обоих стандартизованных коэффициентов смертности.
    if SCR_1 == 0 and SCR_2 == 0:
        return '='

    if SCR_1 > SCR_2:
        if LCL_1 > UCL_2:
            return '>'
        else:
            return '='
    if SCR_1 < SCR_2:
        if UCL_1 < LCL_2:
            return '<'
        else:
            return '='


# In[18]:


def d2d_comparision(df: pd.DataFrame,
                    year: int,
                    district_1: str,
                    district_2: str,
                    param: list,
                    alpha: float = 0.05):
    '''
    Создаёт два среза для сопоставления и передаёт их в функцию comparison.
    df: исходная таблица
    year: год
    district_1: муниципалитет 1
    district_2: муниципалитет 2
    param: как минимум одно значение из набора полов или из набора причина смерти
    alpha: уровень значимости
    Возвращает статистически значимое отличие SCR_1 от SCR_2 в формате '>', '=', '<' или None
    в случае незначимости данных по хотябы одной группе.
    '''
    if year is None:
        slice_1 = df[(df['Муниципалитет']==district_1)&((df['Пол'].isin(param))|(df['Причина смерти'].isin(param)))]
        slice_2 = df[(df['Муниципалитет']==district_2)&((df['Пол'].isin(param))|(df['Причина смерти'].isin(param)))]
    else:
        slice_1 = df[(df['Год']==year)&(df['Муниципалитет']==district_1)&((df['Пол'].isin(param))|(df['Причина смерти'].isin(param)))]
        slice_2 = df[(df['Год']==year)&(df['Муниципалитет']==district_2)&((df['Пол'].isin(param))|(df['Причина смерти'].isin(param)))]
    N_1, D_1, SCR_1, LCL_1, UCL_1, significance_1 =  N_D_SCR_CI_calc(slice_1, alpha)
    N_2, D_2, SCR_2, LCL_2, UCL_2, significance_2 =  N_D_SCR_CI_calc(slice_2, alpha)
    return comparison(SCR_1, LCL_1, UCL_1, SCR_2, LCL_2, UCL_2)


# In[19]:


def in_district_comparision(df: pd.DataFrame,
                            year: int,
                            district: str,
                            param_1: list,
                            param_2: list,
                            alpha: float = 0.05):
    '''
    Создаёт два среза для сопоставления и передаёт их в функцию comparison.
    df: исходная таблица
    year: год
    district: муниципалитет
    param_1: как минимум одно значение из набора полов или из набора причина смерти
    param_2: как минимум одно значение из набора полов или из набора причина смерти
    alpha: уровень значимости
    Возвращает статистически значимое отличие SCR_1 от SCR_2 в формате '>', '=', '<' или None
    в случае незначимости данных по хотябы одной группе.
    '''
    if year is None:
        slice_1 = df[(df['Муниципалитет']==district)&((df['Пол'].isin(param_1))|(df['Причина смерти'].isin(param_1)))]
        slice_2 = df[(df['Муниципалитет']==district)&((df['Пол'].isin(param_2))|(df['Причина смерти'].isin(param_2)))]
    else:
        slice_1 = df[(df['Год']==year)&(df['Муниципалитет']==district)&((df['Пол'].isin(param_1))|(df['Причина смерти'].isin(param_1)))]
        slice_2 = df[(df['Год']==year)&(df['Муниципалитет']==district)&((df['Пол'].isin(param_2))|(df['Причина смерти'].isin(param_2)))]
    N_1, D_1, SCR_1, LCL_1, UCL_1, significance_1 =  N_D_SCR_CI_calc(slice_1, alpha)
    N_2, D_2, SCR_2, LCL_2, UCL_2, significance_2 =  N_D_SCR_CI_calc(slice_2, alpha)
    return comparison(SCR_1, LCL_1, UCL_1, SCR_2, LCL_2, UCL_2)


# ## Вычисление

# ### Статика по годам

# #### Статика между муниципалитетами по годам

print('Статика между муниципалитетами по годам', '\n', '\n')

# In[20]:


# Идея: попарно сравнить различия в стандартизованных коэффициентов смертности между разными муниципалитетами
# для всех комбинаций остальных параметров.


# In[21]:


# Создание таблицы срезов для сравнений между муниципалитетами внутри года.
year_list = []
district_1_list = []
district_2_list = []
param_list = []
# Фильтрация по году.
for year in all_years_list:
    # Фильтрация по первому муниципалитету пары.
    for district_1 in all_districts_list:
        # Фильтрация по второму муниципалитету пары.
        for district_2 in all_districts_list:
            # Защита от повторов вида [a, b] и [b, a].
            if district_2 > district_1:
                # Создание списка, куда будут записываться непрошедшие проверку по числу умерших, причины смерти.
                drop_cause_list = []
                # Проверка по чилу умерших для всех причин в паре.
                if df[(df['Муниципалитет']==district_1)&(df['Год']==year)]['Число умерших'].sum()>9 and \
                df[(df['Муниципалитет']==district_2)&(df['Год']==year)]['Число умерших'].sum()>9:
                    # Добавление значений года, муниципалитетов и полов в списки при успешном прохождениии проверки.
                    year_list.append(year)
                    district_1_list.append(district_1)
                    district_2_list.append(district_2)
                    param_list.append(all_genders_list)
                    # Фильтрация по полу.
                    for gender in all_genders_list:
                        # Приверка по числу смертей.
                        if df[(df['Муниципалитет']==district_1)&(df['Пол']==gender)&(df['Год']==year)]['Число умерших'].sum()>9 and \
                        df[(df['Муниципалитет']==district_2)&(df['Пол']==gender)&(df['Год']==year)]['Число умерших'].sum()>9:
                            # Добавление значений года, муниципалитетов и пола в списки при успешном прохождениии проверки.
                            year_list.append(year)
                            district_1_list.append(district_1)
                            district_2_list.append(district_2)
                            param_list.append(gender)
                    # Фильтрация по причине смерти.
                    for cause in all_causes_list:
                        # Проверка по числу смертей.
                        if df[(df['Муниципалитет']==district_1)&(df['Причина смерти']==cause)&
                            (df['Год']==year)]['Число умерших'].sum()>9 and \
                        df[(df['Муниципалитет']==district_2)&(df['Причина смерти']==cause)&
                            (df['Год']==year)]['Число умерших'].sum()>9:
                            # Добавление значений года, муниципалитетов и причины смерти в списки при успешном прохождениии проверки.
                            year_list.append(year)
                            district_1_list.append(district_1)
                            district_2_list.append(district_2)
                            param_list.append(cause)
                        else:
                            # Добавление причины смерти в drop_cause_list в случае непрохождения проверки.
                            drop_cause_list.append(cause)
                    # Проверка на наличие причин смерти в drop_cause_list.
                    if len(drop_cause_list) != 0:
                        # Проверка по числу смертей для всех не прошедших предущие проверки причин смерти вместе.
                        if df[(df['Муниципалитет']==district_1)&(df['Причина смерти'].isin(drop_cause_list))&
                            (df['Год']==year)]['Число умерших'].sum()>9 and \
                        df[(df['Муниципалитет']==district_2)&(df['Причина смерти'].isin(drop_cause_list))&
                            (df['Год']==year)]['Число умерших'].sum()>9:
                            # Добавление значений года, муниципалитетов и причин смерти в списки при успешном прохождениии проверки.
                            year_list.append(year)
                            district_1_list.append(district_1)
                            district_2_list.append(district_2)
                            param_list.append(drop_cause_list)
# Создание таблицы slices_d2d_df.
slices_d2d_df = pd.DataFrame({'Год': year_list, 'Муниципалитет 1': district_1_list,
                              'Муниципалитет 2': district_2_list, 'Параметр': param_list})
# Приведение всех данных колонки 'Параметр' к формату list.
slices_d2d_df['Параметр'] = slices_d2d_df['Параметр'].apply(lambda x: [x] if type(x) != list else x)

# Создание колонки 'Значимое отличие (p < alpha)', содержащей значения '>', '=', '<'.
# '>' значит, что СКС для Параметра в Муниципалитете 1 статистически значимо больше, чем такое же значение СКС в Муниципалитете 2.
# '<' значит, что СКС Муниципалитета 1 меньше СКС Муниципалитета 2 по Параметру. '=' означает, что СКС значимо не различаются.
slices_d2d_df['Значимое отличие (p < {})'.format(alpha)] = slices_d2d_df.apply(
    lambda x: d2d_comparision(df, x['Год'], x['Муниципалитет 1'], x['Муниципалитет 2'], x['Параметр']), axis = 1)
# Приведение всех данных колонки 'Параметр' к формату str.
slices_d2d_df['Параметр'] = slices_d2d_df['Параметр'].apply(
    lambda x: x if type(x)!=list else
    x[0] if len(x)==1 else
    'Все' if all([gender in x for gender in all_genders_list])
    else '|'.join(x))
# Сортировка строк slices_d2d_df.
slices_d2d_df.sort_values(by=['Муниципалитет 1', 'Муниципалитет 2', 'Параметр', 'Год'], ignore_index=True, inplace=True)


# In[22]:


print('slices_d2d_df', '\n')
print(slices_d2d_df.to_string(), '\n', '\n')


# In[23]:


# Проверка содержимого колонки 'Значимое отличие (p < alpha)'
slices_d2d_df['Значимое отличие (p < {})'.format(alpha)].unique()


# In[24]:


# Словарь замен для расчёта направленности общего различия между группами.
p_dict = {'=': 0, '>': 1, '<': -1}


# In[25]:


# Создание колонки "Число значимых измерений" (показывает после схлопывания по годам сколько пар в группе были значимы).
slices_d2d_df['Число значимых измерений'] = \
slices_d2d_df['Значимое отличие (p < {})'.format(alpha)].apply(lambda x: 0 if x is None else 1)
# Создание колонки "Значимое отличие (p < 0.05) int" (где +1 соответствует '>', 0 соответствует '=', -1 соответствует '<').
# Колонка "Значимое отличие (p < 0.05) int" не была сгенерирована сразу т.к. знаки '>', '=', '<' более наглядные.
slices_d2d_df['Значимое отличие (p < {} int)'.format(alpha)] = \
slices_d2d_df['Значимое отличие (p < {})'.format(alpha)].apply(lambda x: p_dict[x])
# Суммирование по колонкам "Число значимых измерений" и "Значимое отличие (p < 0.05) int" внутри каждой группы (схлопывание по годам).
print('slices_d2d_df.groupby', '\n')
print(slices_d2d_df.groupby(['Муниципалитет 1', 'Муниципалитет 2', 'Параметр'], as_index=False)
    [['Значимое отличие (p < {} int)'.format(alpha), 'Число значимых измерений']].sum().to_string(), '\n', '\n')


# In[26]:


# Короткий вывод:
# 1) Район 1 и Район 2 не отличаются по структуре смертности.
# 2) В Районе 3 женская смертность, смерность ото всех причин и смертность от болезней сердца ниже, чем в Районе 1.
# 3) В Районе 3 смертность от болезней сердца ниже чем в Районе 2.


# #### Статика внутри муниципалитетов по годам

print('Статика внутри муниципалитетов по годам', '\n', '\n')

# In[27]:


# Идея: попарно сравнить различия в стандартизованных коэффициентов смертности между гендерами и причинами смерти
# для всех комбинаций остальных параметров.


# In[28]:


# Создание таблицы срезов для муниципалитета и года.
year_list = []
district_list = []
param_list = []
# Фильтрация по году.
for year in all_years_list:
    # Фильтрация по муниципалитету.
    for district in all_districts_list:
        # Создание списка, куда будут записываться непрошедшие проверку по числу умерших, причины смерти.
        drop_cause_list = []
        # Проверка по чилу умерших для всех причин.
        if df[(df['Муниципалитет']==district)&(df['Год']==year)]['Число умерших'].sum()>9:
            # Добавление значений года, муниципалитета в списки при успешном прохождениии проверки.
            year_list.append(year)
            district_list.append(district)
            param_list.append('Все')
            # Фильтрация по полу.
            for gender in all_genders_list:
                # Проверка по чилу умерших для каждого пола.
                if df[(df['Муниципалитет']==district)&(df['Пол']==gender)&(df['Год']==year)]['Число умерших'].sum()>9:
                    # Добавление значений года, муниципалитета и пола в списки при успешном прохождениии проверки.
                    year_list.append(year)
                    district_list.append(district)
                    param_list.append(gender)
            # Фильтрация по причине смерти.
            for cause in all_causes_list:
                # Проверка по чилу умерших для каждой причины смерти.
                if df[(df['Муниципалитет']==district)&(df['Причина смерти']==cause)&(df['Год']==year)]['Число умерших'].sum()>9:
                    # Добавление значений года, муниципалитета и причины смерти в списки при успешном прохождениии проверки.
                    year_list.append(year)
                    district_list.append(district)
                    param_list.append(cause)
                else:
                    # Добавление причины смерти в drop_cause_list в случае непрохождения проверки.
                    drop_cause_list.append(cause)
            # Проверка на наличие причин смерти в drop_cause_list
            if len(drop_cause_list) != 0:
                # Проверка по числу смертей для всех не прошедших предущие проверки причин смерти вместе.
                if df[(df['Муниципалитет']==district)&(df['Причина смерти'].isin(drop_cause_list))&(df['Год']==year)]['Число умерших'].sum()>9:
                    # Добавление значений года, муниципалитетов и причин смерти в списки при успешном прохождениии проверки.
                    year_list.append(year)
                    district_list.append(district)
                    param_list.append(drop_cause_list)
# Создание таблицы df_per_district.
df_per_district = pd.DataFrame({'Год': year_list, 'Муниципалитет': district_list, 'Параметр': param_list})

# Создание таблицы срезов для попарных сравнений внутри муниципалитета и года.
year_list = []
district_list = []
param_1_list = []
param_2_list = []
# Фильтрация по году.
for year in all_years_list:
    # Фильтрация по муниципалитету.
    for district in all_districts_list:
        # Создание списков по всем параметрам (пол + причины смерти) и отдельно по причинам смерти.
        param_slice_list = df_per_district[(df_per_district['Муниципалитет']==district)&(df_per_district['Год']==year)]['Параметр'].tolist()
        cause_slice_list = [x for x in param_slice_list if x not in (['Все'] + all_genders_list)]
        # Проверка на наличие обоих полов в списке для сравнения М/Ж.
        if all_genders_list[0] in param_slice_list and all_genders_list[1] in param_slice_list:
            # Добавление значений года, муниципалитета и полов в списки при успешном прохождениии проверки.
            year_list.append(year)
            district_list.append(district)
            param_1_list.append(all_genders_list[0])
            param_2_list.append(all_genders_list[1])
        # Проверка на наличе хотя бы двух причин смерти в списке для сравнения причина 1/причина 2.
        if len(cause_slice_list) > 1:
            # Фильтрация по причинам.
            for cause_1 in cause_slice_list:
                for cause_2 in cause_slice_list:
                    # Проверка на то, что обе причины формата str.
                    if type(cause_1) == str and type(cause_2) == str:
                        # Защита от повторов вида [a, b] и [b, a].
                        if cause_1 > cause_2:
                            # Добавление значений года, муниципалитета и причин смерти в списки при успешном прохождениии проверки.
                            year_list.append(year)
                            district_list.append(district)
                            param_1_list.append(cause_1)
                            param_2_list.append(cause_2)
                    else:
                        # Защита от повторов вида [a, b] и [b, a].
                        if type(cause_1) == str:
                            # Добавление значений года, муниципалитета и причин смерти в списки при успешном прохождениии проверки.
                            year_list.append(year)
                            district_list.append(district)
                            param_1_list.append(cause_1)
                            param_2_list.append(cause_2)

# Создание таблицы df_pair_in_district.
df_pair_in_district = pd.DataFrame({'Год': year_list, 'Муниципалитет': district_list, 'Параметр 1': param_1_list, 'Параметр 2': param_2_list,})
# Приведение всех данных колонки 'Параметр 1' к формату list.
df_pair_in_district['Параметр 1'] = df_pair_in_district['Параметр 1'].apply(lambda x: [x] if type(x) != list else x)
# Приведение всех данных колонки 'Параметр 2' к формату list.
df_pair_in_district['Параметр 2'] = df_pair_in_district['Параметр 2'].apply(lambda x: [x] if type(x) != list else x)

# Создание колонки 'Значимое отличие (p < alpha)', содержащей значения '>', '=', '<'.
# '>' значит, что СКС для Параметра 1 статистически значимо больше, чем значение СКС для Параметра 2.
# '<' значит, что СКС для Параметра 1 меньше СКС для Параметра 2. '=' означает, что СКС значимо не различаются.
df_pair_in_district['Значимое отличие (p < {})'.format(alpha)] = df_pair_in_district.apply(
    lambda x: in_district_comparision(df, x['Год'], x['Муниципалитет'], x['Параметр 1'], x['Параметр 2']), axis = 1)
# Приведение всех данных колонки 'Параметр 1' к формату str.
df_pair_in_district['Параметр 1'] = df_pair_in_district['Параметр 1'].apply(
    lambda x: x if type(x)!=list else
    x[0] if len(x)==1 else
    '|'.join(x))
# Приведение всех данных колонки 'Параметр 2' к формату str.
df_pair_in_district['Параметр 2'] = df_pair_in_district['Параметр 2'].apply(
    lambda x: x if type(x)!=list else
    x[0] if len(x)==1 else
    '|'.join(x))
# Сортировка строк df_pair_in_district.
df_pair_in_district.sort_values(by=['Муниципалитет', 'Параметр 1', 'Параметр 2', 'Год'], ignore_index=True, inplace=True)


# In[29]:


print('df_pair_in_district', '\n')
print(df_pair_in_district.to_string(), '\n', '\n')


# In[30]:


# Проверка содержимого колонки 'Значимое отличие (p < alpha)'
df_pair_in_district['Значимое отличие (p < {})'.format(alpha)].unique()


# In[31]:


# Словарь замен для расчёта направленности общего различия между группами.
p_dict = {'=': 0, '>': 1, '<': -1}


# In[32]:


# Создание колонки "Число значимых измерений" (показывает после схлопывания по годам сколько пар в группе были значимы).
df_pair_in_district['Число значимых измерений'] = \
df_pair_in_district['Значимое отличие (p < {})'.format(alpha)].apply(lambda x: 0 if x is None else 1)
# Создание колонки "Значимое отличие (p < 0.05) int" (где +1 соответствует '>', 0 соответствует '=', -1 соответствует '<').
# Колонка "Значимое отличие (p < 0.05) int" не была сгенерирована сразу т.к. знаки '>', '=', '<' более наглядные.
df_pair_in_district['Значимое отличие (p < {} int)'.format(alpha)] = \
df_pair_in_district['Значимое отличие (p < {})'.format(alpha)].apply(lambda x: p_dict[x])
# Суммирование по колонкам "Число значимых измерений" и "Значимое отличие (p < 0.05) int" внутри каждой группы (схлопывание по годам).
print('df_pair_in_district.groupby', '\n')
print(df_pair_in_district.groupby(['Муниципалитет', 'Параметр 1', 'Параметр 2'], as_index=False)
    [['Значимое отличие (p < {} int)'.format(alpha), 'Число значимых измерений']].sum().to_string(), '\n', '\n')


# In[33]:


# Короткий вывод:
# 1) В Районах 1 и 3 женская смертность равна мужской.
# 2) В Районе 3 смертность от болезней сердца выше, чем от остальных причин.


# ### Статика за период

# #### Статика между муниципалитетами за период

print('Статика между муниципалитетами за период', '\n', '\n')

# In[34]:


# Идея: попарно сравнить различия в стандартизованных коэффициентов смертности между разными муниципалитетами
# для всех комбинаций остальных параметров.


# In[35]:


# Создание таблицы срезов для сравнений между муниципалитетами за период.
district_1_list = []
district_2_list = []
param_list = []
# Фильтрация по первому муниципалитету пары.
for district_1 in all_districts_list:
    # Фильтрация по второму муниципалитету пары.
    for district_2 in all_districts_list:
        # Защита от повторов вида [a, b] и [b, a].
        if district_2 > district_1:
            # Создание списка, куда будут записываться непрошедшие проверку по числу умерших, причины смерти.
            drop_cause_list = []
            # Проверка по чилу умерших для всех причин в паре.
            if df[(df['Муниципалитет']==district_1)]['Число умерших'].sum()>9 and \
            df[(df['Муниципалитет']==district_2)]['Число умерших'].sum()>9:
                # Добавление значений муниципалитетов и полов в списки при успешном прохождениии проверки.
                district_1_list.append(district_1)
                district_2_list.append(district_2)
                param_list.append(all_genders_list)
                # Фильтрация по полу.
                for gender in all_genders_list:
                    # Приверка по числу смертей.
                    if df[(df['Муниципалитет']==district_1)&(df['Пол']==gender)]['Число умерших'].sum()>9 and \
                    df[(df['Муниципалитет']==district_2)&(df['Пол']==gender)]['Число умерших'].sum()>9:
                        # Добавление значений муниципалитетов и пола в списки при успешном прохождениии проверки.
                        district_1_list.append(district_1)
                        district_2_list.append(district_2)
                        param_list.append(gender)
                # Фильтрация по причине смерти.
                for cause in all_causes_list:
                    # Проверка по числу смертей.
                    if df[(df['Муниципалитет']==district_1)&(df['Причина смерти']==cause)]['Число умерших'].sum()>9 and \
                    df[(df['Муниципалитет']==district_2)&(df['Причина смерти']==cause)]['Число умерших'].sum()>9:
                        # Добавление значений муниципалитетов и причины смерти в списки при успешном прохождениии проверки.
                        district_1_list.append(district_1)
                        district_2_list.append(district_2)
                        param_list.append(cause)
                    else:
                        # Добавление причины смерти в drop_cause_list в случае непрохождения проверки.
                        drop_cause_list.append(cause)
                # Проверка на наличие причин смерти в drop_cause_list.
                if len(drop_cause_list) != 0:
                    # Проверка по числу смертей для всех не прошедших предущие проверки причин смерти вместе.
                    if df[(df['Муниципалитет']==district_1)&(df['Причина смерти'].isin(drop_cause_list))]['Число умерших'].sum()>9 and \
                    df[(df['Муниципалитет']==district_2)&(df['Причина смерти'].isin(drop_cause_list))]['Число умерших'].sum()>9:
                        # Добавление значений муниципалитетов и причин смерти в списки при успешном прохождениии проверки.
                        district_1_list.append(district_1)
                        district_2_list.append(district_2)
                        param_list.append(drop_cause_list)
# Создание таблицы slices_all_years_d2d_df.
slices_all_years_d2d_df = pd.DataFrame({'Муниципалитет 1': district_1_list,
                              'Муниципалитет 2': district_2_list, 'Параметр': param_list})
# Приведение всех данных колонки 'Параметр' к формату list.
slices_all_years_d2d_df['Параметр'] = slices_all_years_d2d_df['Параметр'].apply(lambda x: [x] if type(x) != list else x)

# Создание колонки 'Значимое отличие (p < alpha)', содержащей значения '>', '=', '<'.
# '>' значит, что СКС для Параметра в Муниципалитете 1 статистически значимо больше, чем такое же значение СКС в Муниципалитете 2.
# '<' значит, что СКС Муниципалитета 1 меньше СКС Муниципалитета 2 по Параметру. '=' означает, что СКС значимо не различаются.
slices_all_years_d2d_df['Значимое отличие (p < {})'.format(alpha)] = slices_all_years_d2d_df.apply(
    lambda x: d2d_comparision(df, None, x['Муниципалитет 1'], x['Муниципалитет 2'], x['Параметр']), axis = 1)
# Приведение всех данных колонки 'Параметр' к формату str.
slices_all_years_d2d_df['Параметр'] = slices_all_years_d2d_df['Параметр'].apply(
    lambda x: x if type(x)!=list else
    x[0] if len(x)==1 else
    'Все' if all([gender in x for gender in all_genders_list])
    else '|'.join(x))
# Сортировка строк slices_all_years_d2d_df.
slices_all_years_d2d_df.sort_values(by=['Муниципалитет 1', 'Муниципалитет 2', 'Параметр'], ignore_index=True, inplace=True)


# In[36]:


print('slices_all_years_d2d_df', '\n')
print(slices_all_years_d2d_df.to_string(), '\n', '\n')


# In[37]:


# Короткий вывод:
# 1) Район 1 и Район 2 не отличаются по структуре смертности.
# 2) В Районе 3 женская смертность и смертность от болезней сердца ниже, чем в Районах 1 и 2.


# #### Статика внутри муниципалитетов за период

print('Статика внутри муниципалитетов за период', '\n', '\n')

# In[38]:


# Идея: попарно сравнить различия в стандартизованных коэффициентов смертности между гендерами и причинами смерти
# для всех комбинаций остальных параметров.


# In[39]:


# Создание таблицы срезов для муниципалитета.
district_list = []
param_list = []
# Фильтрация по муниципалитету.
for district in all_districts_list:
    # Создание списка, куда будут записываться непрошедшие проверку по числу умерших, причины смерти.
    drop_cause_list = []
    # Проверка по чилу умерших для всех причин.
    if df[(df['Муниципалитет']==district)]['Число умерших'].sum()>9:
        # Добавление значений муниципалитета в списки при успешном прохождениии проверки.
        district_list.append(district)
        param_list.append('Все')
        # Фильтрация по полу.
        for gender in all_genders_list:
            # Проверка по чилу умерших для каждого пола.
            if df[(df['Муниципалитет']==district)&(df['Пол']==gender)]['Число умерших'].sum()>9:
                # Добавление значений муниципалитета и пола в списки при успешном прохождениии проверки.
                district_list.append(district)
                param_list.append(gender)
        # Фильтрация по причине смерти.
        for cause in all_causes_list:
            # Проверка по чилу умерших для каждой причины смерти.
            if df[(df['Муниципалитет']==district)&(df['Причина смерти']==cause)]['Число умерших'].sum()>9:
                # Добавление значений муниципалитета и причины смерти в списки при успешном прохождениии проверки.
                district_list.append(district)
                param_list.append(cause)
            else:
                # Добавление причины смерти в drop_cause_list в случае непрохождения проверки.
                drop_cause_list.append(cause)
        # Проверка на наличие причин смерти в drop_cause_list
        if len(drop_cause_list) != 0:
            # Проверка по числу смертей для всех не прошедших предущие проверки причин смерти вместе.
            if df[(df['Муниципалитет']==district)&(df['Причина смерти'].isin(drop_cause_list))]['Число умерших'].sum()>9:
                # Добавление значений года, муниципалитетов и причин смерти в списки при успешном прохождениии проверки.
                district_list.append(district)
                param_list.append(drop_cause_list)
# Создание таблицы df_per_district.
df_per_district = pd.DataFrame({'Муниципалитет': district_list, 'Параметр': param_list})

# Создание таблицы срезов для попарных сравнений внутри муниципалитета и года.
district_list = []
param_1_list = []
param_2_list = []
# Фильтрация по муниципалитету.
for district in all_districts_list:
    # Создание списков по всем параметрам (пол + причины смерти) и отдельно по причинам смерти.
    param_slice_list = df_per_district[(df_per_district['Муниципалитет']==district)]['Параметр'].tolist()
    cause_slice_list = [x for x in param_slice_list if x not in (['Все'] + all_genders_list)]
    # Проверка на наличие обоих полов в списке для сравнения М/Ж.
    if all_genders_list[0] in param_slice_list and all_genders_list[1] in param_slice_list:
        # Добавление значений муниципалитета и полов в списки при успешном прохождениии проверки.
        district_list.append(district)
        param_1_list.append(all_genders_list[0])
        param_2_list.append(all_genders_list[1])
    # Проверка на наличе хотя бы двух причин смерти в списке для сравнения причина 1/причина 2.
    if len(cause_slice_list) > 1:
        # Фильтрация по причинам.
        for cause_1 in cause_slice_list:
            for cause_2 in cause_slice_list:
                # Проверка на то, что обе причины формата str.
                if type(cause_1) == str and type(cause_2) == str:
                    # Защита от повторов вида [a, b] и [b, a].
                    if cause_1 > cause_2:
                        # Добавление значений муниципалитета и причин смерти в списки при успешном прохождениии проверки.
                        district_list.append(district)
                        param_1_list.append(cause_1)
                        param_2_list.append(cause_2)
                else:
                    # Защита от повторов вида [a, b] и [b, a].
                    if type(cause_1) == str:
                        # Добавление значений муниципалитета и причин смерти в списки при успешном прохождениии проверки.
                        district_list.append(district)
                        param_1_list.append(cause_1)
                        param_2_list.append(cause_2)

# Создание таблицы df_pair_all_years_in_district.
df_pair_all_years_in_district = pd.DataFrame({'Муниципалитет': district_list, 'Параметр 1': param_1_list, 'Параметр 2': param_2_list,})
# Приведение всех данных колонки 'Параметр 1' к формату list.
df_pair_all_years_in_district['Параметр 1'] = df_pair_all_years_in_district['Параметр 1'].apply(lambda x: [x] if type(x) != list else x)
# Приведение всех данных колонки 'Параметр 2' к формату list.
df_pair_all_years_in_district['Параметр 2'] = df_pair_all_years_in_district['Параметр 2'].apply(lambda x: [x] if type(x) != list else x)

# Создание колонки 'Значимое отличие (p < alpha)', содержащей значения '>', '=', '<'.
# '>' значит, что СКС для Параметра 1 статистически значимо больше, чем значение СКС для Параметра 2.
# '<' значит, что СКС для Параметра 1 меньше СКС для Параметра 2. '=' означает, что СКС значимо не различаются.
df_pair_all_years_in_district['Значимое отличие (p < {})'.format(alpha)] = df_pair_all_years_in_district.apply(
    lambda x: in_district_comparision(df, None, x['Муниципалитет'], x['Параметр 1'], x['Параметр 2']), axis = 1)
# Приведение всех данных колонки 'Параметр 1' к формату str.
df_pair_all_years_in_district['Параметр 1'] = df_pair_all_years_in_district['Параметр 1'].apply(
    lambda x: x if type(x)!=list else
    x[0] if len(x)==1 else
    '|'.join(x))
# Приведение всех данных колонки 'Параметр 2' к формату str.
df_pair_all_years_in_district['Параметр 2'] = df_pair_all_years_in_district['Параметр 2'].apply(
    lambda x: x if type(x)!=list else
    x[0] if len(x)==1 else
    '|'.join(x))
# Сортировка строк df_pair_all_years_in_district.
df_pair_all_years_in_district.sort_values(by=['Муниципалитет', 'Параметр 1', 'Параметр 2'], ignore_index=True, inplace=True)


# In[40]:


print('df_pair_all_years_in_district', '\n')
print(df_pair_all_years_in_district.to_string(), '\n', '\n')


# In[41]:


# Короткий вывод:
# 1) В Районах 1, 2 и 3 смертность от болезней сердца превышает смертность от остальных причин.
# 2) Районах 1 и 2 женская смертность не отличается от мужской.
# 3) В Районе 3 женская смертность ниже мужской.


# # Выводы

# In[42]:


print('''
# Между муниципалитетами по годам:
# 1) Район 1 и Район 2 не отличаются по структуре смертности.
# 2) В Районе 3 женская смертность, смерность ото всех причин и смертность от болезней сердца ниже, чем в Районе 1.
# 3) В Районе 3 смертность от болезней сердца ниже чем в Районе 2.
# Между муниципалитетами за период:
# 1) Район 1 и Район 2 не отличаются по структуре смертности.
# 2) В Районе 3 женская смертность и смертность от болезней сердца ниже, чем в Районах 1 и 2.
# и смертность от других болезней органов дыхания выше в Районе 3, чем в районе 1. Что говорит об отличной от Районов 1 и 2 структуре смертности.
# Внутри муниципалитетов по годам:
# 1) В Районах 1 и 3 женская смертность равна мужской.
# 2) В Районе 3 смертность от болезней сердца выше, чем от остальных причин.
# Внутри муниципалитетов за период:
# 1) В Районах 1, 2 и 3 смертность от болезней сердца превышает смертность от остальных причин.
# 2) Районах 1 и 2 женская смертность не отличается от мужской.
# 3) В Районе 3 женская смертность ниже мужской.
''')
