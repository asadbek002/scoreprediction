import numpy as np
import matplotlib.pyplot as plt

def load_data(filename):
    data = np.loadtxt(filename, delimiter=',')
    return data

def predict_final_scores(midterm_scores, slope, y_intercept):
    return slope * midterm_scores + y_intercept

if __name__ == '__main__':
    # Загрузка данных
    class_kr = load_data('data/class_score_kr.csv')
    class_en = load_data('data/class_score_en.csv')
    data = np.vstack((class_kr, class_en))

    # Выполнение линейной регрессии для нахождения параметров линии наилучшего подгонки
    slope, y_intercept = np.polyfit(data[:, 0], data[:, 1], 1)

    # Создание списка для хранения предсказанных оценок
    predicted_scores = []

    # Предсказание оценок и сохранение результатов
    for midterm_score in data[:, 0]:
        final_score = predict_final_scores(midterm_score, slope, y_intercept)
        predicted_scores.append((midterm_score, final_score))

    # Печать предсказанных данных
    for midterm_score, final_score in predicted_scores:
        print(f'For midterm score {midterm_score:.2f}, predicted final score is {final_score:.2f}')

    # Построение графика
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'r.', label='Исходные данные')
    plt.plot(data[:, 0], predict_final_scores(data[:, 0], slope, y_intercept), 'b-', label='Предсказание')
    plt.xlabel('Баллы за промежуточный экзамен')
    plt.ylabel('Баллы за экзамен')
    plt.grid()
    plt.legend()
    plt.show()
