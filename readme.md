# Sudoku Solver

## 실행 순서
- python/create_training.py 실행
  - 훈련데이터 trainingdata.txt, traininglable.txt 생성됨.
  - 훈련에 쓰인 데이터
    - mnist dataset (jpg) (용량이 너무 커서 첨부파일에 넣지 않음)
    - 인쇄된 숫자 (jpg) (training 폴더)

- python/solver.py 실행
  - ./python/sudoku_img/sudoku.jpg 사용함
  - 숫자 인식을 위한 testingdata.txt 생성됨
  - 스도쿠 KNN 결과 sudoku_input.txt 생성됨

- python/sudoku_algorithm.py 실행
  - 인식이 성공하여 sudoku_input.txt가 제대로 된 스도쿠 문제라면, sudoku_output.txt 생성됨

- python/sudoku_result.py 실행
  - 스도쿠 사진은 solver.py에서 사용한 이미지.
  - sudoku_output.txt가 제대로 생성되었다면 인식한 스도쿠사진에 스도쿠 결과가 오버레이 되어 출력

#### 원본 스도쿠 사진
![sudoku.jpg](https://github.com/idjoopal/sudoku_solver/blob/master/python/sudoku_img/sudoku.jpg)

#### solver.py - 스도쿠 경계인식
![sudoku.jpg](https://github.com/idjoopal/sudoku_solver/blob/master/md/example1.png)

#### solver.py - 스도쿠 추출
![sudoku.jpg](https://github.com/idjoopal/sudoku_solver/blob/master/md/example2.png)

#### solver.py - 숫자 인식 결과(KNN 분류 알고리즘) - 정확도 떨어짐
![sudoku.jpg](https://github.com/idjoopal/sudoku_solver/blob/master/md/example3.png)

#### sudoku_result.py - 만일 숫자 인식이 잘됐을 경우, 스도쿠 풀이 결과
![sudoku.jpg](https://github.com/idjoopal/sudoku_solver/blob/master/md/example4.png)
