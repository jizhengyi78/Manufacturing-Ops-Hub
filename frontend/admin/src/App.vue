<template>
  <el-container class="app">
    <el-header class="header">
      <h2>制造业知识库管理后台</h2>
      <el-tag type="info">{{ stats }}</el-tag>
    </el-header>

    <el-main>
      <el-tabs v-model="activeTab" @tab-change="onTabChange">
        <!-- Tab 1: 文档列表 -->
        <el-tab-pane label="文档列表" name="list">
          <div class="toolbar">
            <el-input v-model="searchTitle" placeholder="搜索文档标题..." style="width:300px" clearable @input="filterDocs" />
            <el-button type="danger" plain @click="refreshList" :icon="RefreshRight" style="margin-left:8px">刷新</el-button>
          </div>
          <el-table :data="filteredDocs" border stripe style="margin-top:12px" v-loading="loading">
            <el-table-column prop="title" label="文档名称" min-width="250" />
            <el-table-column prop="chunks" label="分块数" width="80" align="center" />
            <el-table-column prop="equipment_model" label="关联设备" width="160" />
            <el-table-column label="操作" width="100" align="center">
              <template #default="{ row }">
                <el-button type="danger" size="small" text @click="deleteDoc(row.title)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <!-- Tab 2: 上传 -->
        <el-tab-pane label="上传文档" name="upload">
          <el-card>
            <el-form label-width="100px">
              <el-form-item label="选择文件">
                <el-upload
                  :auto-upload="false"
                  :on-change="onFileChange"
                  :file-list="uploadFiles"
                  :limit="10"
                  multiple
                  accept=".txt,.md,.csv"
                  drag
                >
                  <el-icon class="el-icon--upload" :size="32"><UploadFilled /></el-icon>
                  <div class="el-upload__text">拖拽文件到此处或 <em>点击选择</em></div>
                  <template #tip><div class="tip">支持 PDF/Word/Excel/TXT/MD/CSV，可一次选择多个文件</div></template>
                </el-upload>
              </el-form-item>
              <el-form-item label="文档类型">
                <el-select v-model="uploadType" style="width:200px">
                  <el-option label="设备手册" value="equipment_manual" />
                  <el-option label="SOP文档" value="sop" />
                  <el-option label="故障案例" value="fault_case" />
                  <el-option label="质检标准" value="quality_standard" />
                  <el-option label="报警码" value="alarm_code" />
                </el-select>
              </el-form-item>
              <el-form-item label="关联设备">
                <el-input v-model="uploadModel" placeholder="如: 海天MA1200" style="width:200px" />
              </el-form-item>
              <el-form-item v-if="uploadFiles.length">
                <el-button type="success" @click="doUpload" :loading="uploading">
                  上传索引 ({{ uploadFiles.length }} 个文件)
                </el-button>
                <el-button text @click="uploadFiles=[]">清空</el-button>
              </el-form-item>
            </el-form>
            <el-alert v-if="uploadResult" :title="uploadResult" :type="uploadError ? 'error' : 'success'" closable @close="uploadResult=''" />
          </el-card>
        </el-tab-pane>

        <!-- Tab 3: 搜索测试 -->
        <el-tab-pane label="搜索测试" name="search">
          <div class="toolbar">
            <el-input v-model="searchQuery" placeholder="输入检索词测试..." style="width:400px" @keyup.enter="doSearch" />
            <el-button type="primary" @click="doSearch" style="margin-left:8px">搜索</el-button>
            <span v-if="searchLatency" style="margin-left:12px;color:#909399">{{ searchLatency }}ms</span>
          </div>
          <el-table :data="searchResults" border stripe style="margin-top:12px" v-if="searchResults.length">
            <el-table-column label="排名" width="60" align="center"><template #default="{ $index }">{{ $index + 1 }}</template></el-table-column>
            <el-table-column prop="content_preview" label="内容预览" min-width="400" />
            <el-table-column prop="relevance_score" label="相关性" width="90" align="center" />
            <el-table-column prop="source" label="来源" width="70" align="center" />
          </el-table>
          <el-empty v-else-if="searched" description="无结果" />
        </el-tab-pane>
      </el-tabs>
    </el-main>
  </el-container>
</template>

<script setup>
import { ref, computed } from 'vue'

const activeTab = ref('list')
const loading = ref(false)
const docs = ref([])
const searchTitle = ref('')
const filteredDocs = computed(() => {
  if (!searchTitle.value) return docs.value
  return docs.value.filter(d => d.title.toLowerCase().includes(searchTitle.value.toLowerCase()))
})
const stats = computed(() => `${docs.value.length} 份文档`)

// 文档列表
async function refreshList() {
  loading.value = true
  try {
    const r = await fetch('/api/v1/knowledge/documents')
    const j = await r.json()
    docs.value = j.data?.items || []
  } finally { loading.value = false }
}

// 上传
const uploadFiles = ref([])
const uploadType = ref('equipment_manual')
const uploadModel = ref('')
const uploading = ref(false)
const uploadResult = ref('')
const uploadError = ref(false)

function onFileChange(file) {
  const raw = file.raw || file
  if (!uploadFiles.value.some(f => f.name === raw.name)) {
    uploadFiles.value.push(raw)
  }
}

async function doUpload() {
  if (!uploadFiles.value.length) return
  uploading.value = true
  let success = 0, fail = 0
  for (const file of uploadFiles.value) {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('doc_type', uploadType.value)
    fd.append('equipment_model', uploadModel.value)
    try {
      const r = await fetch('/api/v1/knowledge/upload', { method: 'POST', body: fd })
      const j = await r.json()
      if (j.code === 0) { success++ }
      else { fail++ }
    } catch(e) { fail++ }
  }
  uploadResult.value = `上传完成: ${success} 成功, ${fail} 失败`
  uploadError.value = fail > 0
  uploading.value = false
  uploadFiles.value = []
  refreshList()
}

// 搜索测试
const searchQuery = ref('')
const searchResults = ref([])
const searchLatency = ref(0)
const searched = ref(false)

async function doSearch() {
  if (!searchQuery.value.trim()) return
  const r = await fetch(`/api/v1/knowledge/search?query=${encodeURIComponent(searchQuery.value)}&top_k=10`, { method: 'POST' })
  const j = await r.json()
  searchResults.value = j.data?.results || []
  searchLatency.value = j.data?.latency_ms || 0
  searched.value = true
}

// 删除
async function deleteDoc(title) {
  try {
    await fetch(`/api/v1/knowledge/documents/${encodeURIComponent(title)}`, { method: 'DELETE' })
    refreshList()
  } catch(e) { /* ignore */ }
}

function onTabChange(tab) { if (tab === 'list') refreshList() }
refreshList()
</script>

<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
.app { height:100vh; display:flex; flex-direction:column; }
.header { display:flex; align-items:center; justify-content:space-between; background:#409EFF; color:white; height:56px; padding:0 24px; }
.header h2 { font-size:18px; }
.toolbar { display:flex; align-items:center; margin-bottom:8px; }
.tip { font-size:12px; color:#909399; margin-top:4px; }
</style>
