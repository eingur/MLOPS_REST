# MLOPS_REST
## ДЗ.1 и ДЗ.2 по MLOPS

Реализовать API (REST либо процедуры gRPC), которое умеет: 
1. Обучать ML-модель с возможностью настройки 
гиперпараметров. При этом гиперпараметры для разных 
моделей могут быть разные. Минимальное количество классов 
моделей доступных для обучения == 2. 
2. Возвращать список доступных для обучения классов моделей 
3. Возвращать предсказание конкретной модели (как следствие, 
система должна уметь хранить несколько обученных моделей) 
4. Обучать заново и удалять уже обученные модели


### Пример использования
конструктор докер образа 
 
 docker build . -t eingur/mlops_ml:0.1
 
 docker-compose up -d
 
 хост: http://127.0.0.1:5000
 
 докерхаб:https://hub.docker.com/repository/docker/eingur/mlops_ml

- Предикт

curl -X POST -H 'Content-Type: application/json' -d @test.json http://127.0.0.1:5000/prediction/1

curl -X POST -H 'Content-Type: application/json' -d @test.json http://127.0.0.1:5000/prediction/2
- Переобучить с параметрами

curl -X POST -H 'Content-Type: application/json' -d @train.json http://127.0.0.1:5000/refit/2

- удалить модельку

curl -X DELETE http://127.0.0.1:5000/1

curl -X DELETE http://127.0.0.1:5000/2

- получить пул моделек

curl -X GET http://127.0.0.1:5000/
