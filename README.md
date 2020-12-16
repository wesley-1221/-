<div align='center' ><font size='70'>说明文档</font></div>

## 第一次将文件上传到GitHub

- 进入管理的文件夹

- 打开git bash

- 初始化命令

  ```python
  git init
  ```

- 查看目录下文件状态

  ```python
  git status
  
  # 新增文件或者修改过的文件是红色的
  ```

- 管理指定文件（红变绿）

  ```python
  git add 文件名
  git add .      # 所有文件
  ```

- 个人信息配置：用户名,邮箱(只要一次)

  ```python
  git config --global user.email "you@example.com"
  git config --global user.name "your name"
  ```

- 生成版本

  ```python
  git commit -m '描述信息'
  ```

- 查看版本记录

  ```python
  git log
  ```

- 连接GitHub

  ```python
  # 给远程仓库起别名
  git remote add origin 远程仓库地址
  # 向远程仓库推送代码（默认是main分支）
  git push -u origin 分支
  ```

### 第一个版本我已经上传上去了，你们只要将我的这个版本给克隆下来，得到一个文件夹，在这个文件夹里面创建新的文件夹，完成自己任务，避免发生冲突

## 在自己电脑上下载代码

```python
# 克隆远程仓库代码
git clone 远程仓库地址（这个时候不用进行初始化）
# 在将自己完成的任务放到文件夹中
# 完成任务后
# 我的代码已经有了两个分支，克隆下来默认在主分支，我们需要切换分支，进入第二分支，我的第二分支是dev分支
git checkout dev
git status
git add .
git commit -m "描述信息"
# 完成任务后，将代码推送到远端
# 注意只需要推送dev分支，看清楚自己是在哪个分支
git push origin dev
# 推送完成后就可以了，剩下我来合并。
```



