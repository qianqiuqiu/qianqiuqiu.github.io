---
title: '{{ replace .File.ContentBaseName "-" " " | title }}'
# slug 用于生成短 URL（/p/<slug>/），中文标题务必填英文 slug，避免超长百分号编码链接
slug: ''
date: {{ .Date }}
categories: []
tags: []
draft: true
---
