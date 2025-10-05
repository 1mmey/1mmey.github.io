/**
 * 自动转换 Typora 博客格式到 Astro 博客格式
 *
 * Typora 格式:
 *   blog/
 *     article.md
 *     article/
 *       pic1.jpg
 *
 * Astro 格式:
 *   blog/
 *     article/
 *       index.md
 *       pic1.jpg
 */

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const BLOG_DIR = path.join(__dirname, '..', 'src', 'content', 'blog')

/**
 * 检查是否需要转换
 * @param {string} mdFile - Markdown 文件路径
 * @returns {boolean}
 */
function needsConversion(mdFile) {
  const basename = path.basename(mdFile, '.md')

  // 如果文件名是 index.md，跳过
  if (basename === 'index') {
    return false
  }

  // 所有非 index.md 的 .md 文件都需要转换为文件夹结构
  return true
}

/**
 * 修改 Markdown 中的图片路径
 * @param {string} content - Markdown 内容
 * @param {string} folderName - 原文件夹名称
 * @returns {string} 修改后的内容
 */
function updateImagePaths(content, folderName) {
  // 匹配图片引用：![alt](folderName/image.ext) 或 ![alt](./folderName/image.ext)
  const patterns = [
    new RegExp(`!\\[([^\\]]*)\\]\\(${folderName}/([^)]+)\\)`, 'g'),
    new RegExp(`!\\[([^\\]]*)\\]\\(\\./${folderName}/([^)]+)\\)`, 'g')
  ]

  let updatedContent = content

  patterns.forEach((pattern) => {
    updatedContent = updatedContent.replace(pattern, '![$1](./$2)')
  })

  return updatedContent
}

/**
 * 转换单个博客文章
 * @param {string} mdFile - Markdown 文件路径
 */
function convertBlogPost(mdFile) {
  const basename = path.basename(mdFile, '.md')
  const dirname = path.dirname(mdFile)
  const assetFolder = path.join(dirname, basename)
  const newFolder = path.join(dirname, basename)
  const newMdFile = path.join(newFolder, 'index.md')

  console.log(`\n转换: ${basename}`)

  try {
    // 1. 读取 Markdown 内容
    let content = fs.readFileSync(mdFile, 'utf-8')

    // 2. 更新图片路径
    const updatedContent = updateImagePaths(content, basename)

    // 3. 创建新文件夹（如果不存在）
    if (!fs.existsSync(newFolder)) {
      fs.mkdirSync(newFolder, { recursive: true })
    }

    // 4. 移动资源文件到新文件夹（如果存在同名文件夹）
    if (fs.existsSync(assetFolder) && assetFolder !== newFolder) {
      const files = fs.readdirSync(assetFolder)
      let fileCount = 0
      files.forEach((file) => {
        const srcPath = path.join(assetFolder, file)
        const destPath = path.join(newFolder, file)

        if (fs.statSync(srcPath).isFile()) {
          fs.copyFileSync(srcPath, destPath)
          console.log(`  ✓ 复制: ${file}`)
          fileCount++
        }
      })

      if (fileCount === 0) {
        console.log(`  ℹ 原文件夹为空`)
      }
    } else if (!fs.existsSync(assetFolder)) {
      console.log(`  ℹ 无同名资源文件夹，创建空文件夹`)
    }

    // 5. 写入新的 index.md
    fs.writeFileSync(newMdFile, updatedContent, 'utf-8')
    console.log(`  ✓ 创建: index.md`)

    // 6. 删除原始文件和文件夹
    fs.unlinkSync(mdFile)
    console.log(`  ✓ 删除: ${basename}.md`)

    if (fs.existsSync(assetFolder) && assetFolder !== newFolder) {
      fs.rmSync(assetFolder, { recursive: true, force: true })
      console.log(`  ✓ 删除: ${basename}/ (原资源文件夹)`)
    }

    console.log(`✅ 转换完成: ${basename}`)
    console.log(`  💡 提示: 现在可以添加 thumbnail.png 等图片到文件夹中`)
  } catch (error) {
    console.error(`❌ 转换失败: ${basename}`, error.message)
  }
}

/**
 * 扫描并转换所有需要转换的博客文章
 */
function convertAllBlogs() {
  console.log('开始扫描博客目录...')
  console.log(`目录: ${BLOG_DIR}\n`)

  if (!fs.existsSync(BLOG_DIR)) {
    console.error('❌ 博客目录不存在!')
    return
  }

  const items = fs.readdirSync(BLOG_DIR)
  let convertedCount = 0

  items.forEach((item) => {
    const itemPath = path.join(BLOG_DIR, item)

    // 只处理 .md 文件（排除 index.md）
    if (item.endsWith('.md') && item !== 'index.md') {
      if (needsConversion(itemPath)) {
        convertBlogPost(itemPath)
        convertedCount++
      }
    }
  })

  console.log(`\n========================================`)
  console.log(`转换完成! 共处理 ${convertedCount} 篇文章`)
  console.log(`========================================`)
}

// 执行转换
convertAllBlogs()
