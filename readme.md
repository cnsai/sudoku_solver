# Sudoku Solver

## 실행 순서
- python/create_training.py 실행
  - 훈련데이터 trainingdata.txt, traininglable.txt 생성됨.
  - 훈련에 쓰인 데이터
    - mnist dataset (jpg) (용량이 너무 커서 첨부파일에 넣지 않음)
    - 인쇄된 숫자 (jpg) (training 폴더)

- python/solver.py 실행
  - 스도쿠 사진은 python/sudoku_img에서 선정, 코드변경필요
  - 스도쿠 특징추출한 testingdata.txt 생성됨
  - 스도쿠 KNN 결과 sudoku_input.txt 생성됨

- Debug/sudoku_rocognizer.exe 실행
  - sudoku_input.txt가 제대로 인식을 했다면, sudoku_output.txt 생성됨

- python/sudoku_result.py 실행
  - 스도쿠 사진은 solver.py에서 사용한 이미지, 코드변경필요
  - sudoku_output.txt가 제대로 생성되었다면 인식한 스도쿠사진에 스도쿠 결과가 오버레이 되어 출력
 
