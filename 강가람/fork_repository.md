#### 스터디 해주셔야 할 일



##### 동권님께서 공지에 올려주신 정리 내용을 참고해 [1]~[5]번을 해주세요! 안 되는 부분이 있으면 톡에 말씀해주세요! 다같이 고민해용ㅎㅎ!

[동권님 fork 정리][https://github.com/gwonihan/MulcamD_study/blob/master/Fork%20%EC%82%AC%EC%9A%A9%EB%B2%95.md]



> Fork와 Clone의 차이
>
> 전자는 다른 사람이 파일을 수정할 경우 Repository 주인이 내용을 검토한 뒤 merge합니다. 후자는 다른 사람이 파일을 수정할 경우 바로 수정됩니다. 
>
> 후자의 경우 충돌이 일어날 가능성이 커 수정 내용을 전반적으로 검토할 관리자가 존재하는 Fork를 사용하는 것 같습니다.



##### [1] 그룹 레포지토리 Fork 하기 (동권님 정리 1~6번)



> GroupStudy Url
>
> https://github.com/Younghyeon-Kim/Learning-Crew

1. GroupStudy 레포지토리에 접속

2. Fork 클릭

3. 따라하기

4. 따라하기

5. 컴퓨터 디렉토리에 파일 만들어주기

   * 터미널/콘솔 창에서 GroupStudy 폴더를 생성할 디렉토리로 위치를 바꿔줍니다.

     ex: `cd desktop`

   * `git clone <GroupStudy Url> GroupStudy ` 

     *주의: 본인 계정에 Fork한 GroupStudy 레포지토리 Url을 통해 clone 해야 합니다

6. remote repository 추가

   * `git remote -v` 입력후 `origin`이 출력되는지 확인

   * `git remote add upstream <GroupStudy Url>`

   * `git remote -v` 입력후 `origin`과 `upstream`이 출력되는지 확인

     > upstream은 github 레포지토리의 내용을 컴퓨터 디렉토리에 반영하기 위해 만드는 겁니다 - [4]번에서 설명





##### [2] 개인 branch 만들기 (동권님 정리 7번)

`git branch <이름 초성 소문자> `로 해주시면 구분이 편할 거 같아요!

> 여기서 pull request 해주시면 제가 master branch 로 merge 하는 거예요!





##### [3] 파일 수정하기 (동권님 정리 8번)

제가 레포지토리에 [python 기본 100제][https://codeup.kr/problemsetsol.php?psid=33] 중 첫번째 문제를 올려두었습니다. 



1. 각자 만든 branch 에서 작업해주세요!
2. **CodeUp_6001** 폴더에서
3. **CodeUp_6001_이름.py** 형식으로 파일명을 설정해 작업하신 뒤 저장해주세요
4. `git add .` -  `git commit -m 'add <파일명>'` - `git push origin <개인 branch명>` 순으로 github에 push 해주세요!
5. pull request를 넣어주세요!

6. 넣어주신 뒤 카톡에 말씀해주시면 저도 수정 사항을 merge 해 모든 과정이 정상적으로 이루어지는지 확인하겠습니다.





##### [4] 컴퓨터 디렉토리에 있는 폴더를 git hub와 동기화 하기

1. master branch에서 작업해야 합니다! : `git branch master`
2. `git fetch upstream`
3. `git merge upstream/master`

> 이 명령어들을 입력하면 **컴퓨터 GroupStudy 폴더**에 **github의 GroupStudy 레포지토리**가 반영됩니다.

[참고][https://lifove.tistory.com/54]





##### [5] 피드백 달기 -  [1]~[4]를 수행하고 다시 공지하도록 하겠습니다. 

1. 여러분들이 pull request를 통해 올려주신 코드는 merge 하면서 reviewer를 배정해드릴 겁니다! 
2. commit을 누르면 코드를 view 할 수 있습니다.
3. 코드 각줄 좌측에 +를 누르면 코멘트를 달 수 있는 텍스트 박스가 나타나요.
4. 코멘트를 입력하고 Add single comment 버튼을 눌러주시면 코멘트가 등록됩니다.

[참고][https://devlog-wjdrbs96.tistory.com/231]
