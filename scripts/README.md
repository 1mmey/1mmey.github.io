# 博客格式转换脚本

## 功能说明

这个脚本用于自动将 Typora 格式的博客转换为 Astro 博客模板所需的格式。

**支持两种情况：**

### 情况 1：有图片的文章（Typora 格式）

```
src/content/blog/
  article.md
  article/
    pic1.jpg
    pic2.png
```

Markdown 中的图片引用：

```markdown
![图片描述](article/pic1.jpg)
```

### 情况 2：没有图片的单个文章

```
src/content/blog/
  article.md
```

### 转换后（统一的 Astro 格式）

```
src/content/blog/
  article/
    index.md
    pic1.jpg      # 如果有
    pic2.png      # 如果有
```

Markdown 中的图片引用：

```markdown
![图片描述](./pic1.jpg)
```

**优点：** 即使原来没有图片，转换后也可以随时添加 `thumbnail.png` 作为头图

## 使用方法

### 1. 运行脚本

在项目根目录下运行：

```bash
node scripts/convert-blog-structure.js
```

### 2. 脚本会自动：

- ✅ 扫描 `src/content/blog/` 目录
- ✅ 处理所有 `.md` 文件（除了 `index.md`）
- ✅ 创建同名文件夹
- ✅ 将 `.md` 文件重命名为 `index.md` 并移入文件夹
- ✅ 如果有同名资源文件夹，复制所有图片和文件
- ✅ 自动修改 Markdown 中的图片路径为相对路径 `./xxx.jpg`
- ✅ 删除原始文件和旧的资源文件夹

### 3. 输出示例

```
开始扫描博客目录...
目录: D:\Coding\Blog\src\content\blog

转换: TSCTF-J2024-Immey's-write-up
  ✓ 复制: 1.png
  ✓ 复制: 2.png
  ✓ 复制: 3.png
  ✓ 创建: index.md
  ✓ 删除: TSCTF-J2024-Immey's-write-up.md
  ✓ 删除: TSCTF-J2024-Immey's-write-up/ (原文件夹)
✅ 转换完成: TSCTF-J2024-Immey's-write-up

========================================
转换完成! 共处理 1 篇文章
========================================
```

## 注意事项

⚠️ **运行前建议先备份或提交 Git**

脚本会：

- 删除原始 `.md` 文件
- 删除原始资源文件夹
- 创建新的文件夹结构

虽然脚本会复制所有文件，但为了安全起见，建议先备份。

## 以后写博客的流程

### 方案 A：继续用 Typora（推荐使用脚本）

1. 在 Typora 中正常写作，图片保存在同名文件夹
2. 将 `.md` 文件和文件夹都放入 `src/content/blog/`
3. 运行脚本自动转换格式

### 方案 B：直接使用 Astro 格式（不需要脚本）

1. 在 `src/content/blog/` 下创建文章文件夹，如 `my-article/`
2. 在文件夹中创建 `index.md`
3. 将图片直接放在同一文件夹中
4. 在 Markdown 中使用 `./image.jpg` 引用图片

## 支持的图片引用格式

脚本会自动转换以下格式：

```markdown
![alt](article-name/image.jpg) → ![alt](./image.jpg)
![alt](./article-name/image.jpg) → ![alt](./image.jpg)
```

## 故障排除

### 脚本运行报错

如果提示 `Cannot use import statement outside a module`，请确保 `package.json` 中有：

```json
{
  "type": "module"
}
```

### 图片不显示

检查：

1. 图片文件是否在正确的文件夹中
2. 图片路径是否使用了 `./` 前缀
3. 图片文件名大小写是否正确

### 保留原始文件

如果你想保留原始文件（不删除），可以修改脚本中的删除部分（第 6 步）。
