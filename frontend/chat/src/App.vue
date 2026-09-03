<template>
  <!-- 登录页 -->
  <div v-if="!token" class="login-page">
    <div class="login-card">
      <h2>制造业生产运维助手</h2>
      <p class="login-subtitle">请登录后使用</p>
      <el-form @submit.prevent="login" label-width="0">
        <el-form-item>
          <el-input v-model="loginUsername" placeholder="用户名" size="large" prefix-icon="User" />
        </el-form-item>
        <el-form-item>
          <el-input v-model="loginPassword" placeholder="密码" size="large" type="password" show-password prefix-icon="Lock" @keyup.enter="login" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" size="large" @click="login" :loading="loggingIn" style="width:100%">登 录</el-button>
        </el-form-item>
      </el-form>
      <el-alert v-if="loginError" :title="loginError" type="error" show-icon :closable="false" style="margin-top:12px" />
      <div class="login-hint">
        <p>演示账号：</p>
        <el-table :data="demoUsers" size="small" stripe>
          <el-table-column prop="name" label="用户名" width="160" />
          <el-table-column prop="role" label="角色" />
        </el-table>
        <p style="margin-top:8px;color:#909399" v-if="demoPwTip">{{ demoPwTip }}</p>
      </div>
    </div>
  </div>

  <!-- 聊天页 -->
  <el-container v-else class="app-container">
    <el-header class="app-header">
      <div class="header-left"><h2>制造业生产运维助手</h2></div>
      <div class="header-right">
        <el-tag size="small" type="success" style="margin-right:8px">{{ currentUser.display_name }}</el-tag>
        <el-tag size="small" type="info" style="margin-right:8px">{{ currentUser.roleLabel }}</el-tag>
        <el-select v-model="currentWorkshop" style="width:120px" size="small">
          <el-option label="A车间" value="workshop-a" /><el-option label="B车间" value="workshop-b" />
        </el-select>
        <el-button text size="small" @click="logout" style="margin-left:8px;color:white">退出</el-button>
      </div>
    </el-header>

    <el-container class="main-area">
      <el-main class="chat-area">
        <div class="messages" ref="messagesContainer">
          <div v-if="messages.length===0" class="welcome">
            <el-icon :size="48" color="#409EFF"><ChatDotRound /></el-icon>
            <p>你好，{{ currentUser.display_name }}！我是制造业生产运维助手。</p>
            <p>可以帮你查询设备手册、SOP、报警码，或者问故障处理建议。</p>
          </div>

          <div v-for="(msg,idx) in messages" :key="idx" :class="['message',msg.role]">
            <div class="message-avatar">
              <el-avatar v-if="msg.role==='user'" :size="32" icon="UserFilled" />
              <el-avatar v-else :size="32" style="background:#409EFF"><el-icon><Cpu /></el-icon></el-avatar>
            </div>
            <div class="message-content">
              <div class="message-text" v-html="formatMessage(msg.content)" />
              <img v-if="msg.image" :src="msg.image" style="max-width:240px;border-radius:8px;margin-top:6px;cursor:pointer" @click="previewImage(msg.image)" />
              <div v-if="msg.citations?.length" class="citations">
                <el-tag v-for="(cit,ci) in msg.citations" :key="ci" size="small" type="info" style="margin-right:6px;margin-top:4px">
                  📎 {{ cit.doc_title || cit.chunk_id }}
                </el-tag>
              </div>
              <div v-if="msg.model" class="message-meta">{{ msg.model }} · {{ msg.time }}</div>
            </div>
          </div>

          <div v-if="streaming" class="message assistant">
            <div class="message-avatar"><el-avatar :size="32" style="background:#409EFF"><el-icon><Cpu /></el-icon></el-avatar></div>
            <div class="message-content">
              <div v-if="streamingText" class="message-text">{{ streamingText }}</div>
              <div v-else class="thinking-text"><span class="thinking-dots">智能运维助手正在思考中</span><span class="dot-anim">...</span></div>
            </div>
          </div>
        </div>

        <div class="input-area">
          <div class="quick-actions">
            <el-button size="small" @click="quickAction('查SOP')">📖 查SOP</el-button>
            <el-button size="small" @click="quickAction('查报警码')">⚠ 查报警码</el-button>
            <el-button size="small" @click="quickAction('故障处理')">🔧 故障处理</el-button>
            <el-button size="small" @click="quickAction('设备点检')">✅ 设备点检</el-button>
            <el-button size="small" @click="quickAction('换模步骤')">🔄 换模步骤</el-button>
          </div>
          <div class="input-row">
            <el-button size="large" @click="toggleVoice" :type="listening ? 'warning' : 'default'" :icon="Microphone" circle style="flex-shrink:0" :title="listening ? '正在聆听...' : '语音输入'" />
            <el-upload :auto-upload="false" :show-file-list="false" :on-change="onImageSelect" accept="image/*" style="flex-shrink:0;margin-left:4px">
              <el-button size="large" icon="PictureFilled" circle title="上传图片" />
            </el-upload>
            <el-input v-model="inputText" placeholder="输入问题，或点击🎤语音..." @keyup.enter="sendMessage" :disabled="streaming" size="large" clearable style="margin-left:4px" />
            <el-button type="primary" size="large" @click="sendMessage" :loading="streaming" :disabled="!inputText.trim()" style="margin-left:4px">
              <el-icon><Promotion /></el-icon> 发送
            </el-button>
          </div>
        </div>
      </el-main>

      <el-aside class="info-panel" width="320px">
        <el-card header="📋 当前状态" shadow="never">
          <el-descriptions :column="1" size="small" border>
            <el-descriptions-item label="角色">{{ currentUser.roleLabel }}</el-descriptions-item>
            <el-descriptions-item label="车间">{{ currentWorkshop }}</el-descriptions-item>
            <el-descriptions-item label="会话ID">{{ sessionId?.slice(0,12) || '未开始' }}</el-descriptions-item>
            <el-descriptions-item label="消息数">{{ messages.length }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
        <el-card header="⚙ 快捷设备" shadow="never" style="margin-top:12px">
          <el-tag v-for="eq in equipmentList" :key="eq" size="small" style="margin:4px;cursor:pointer" @click="quickAction(eq)">{{ eq }}</el-tag>
        </el-card>
        <el-card shadow="never" style="margin-top:12px">
          <template #header>
            <div style="display:flex;justify-content:space-between;align-items:center">
              <span>💬 会话历史</span>
              <el-button type="primary" size="small" @click="newSession" :icon="Plus">新建</el-button>
            </div>
          </template>
          <div v-if="!userSessions.length" class="text-secondary">暂无历史会话</div>
          <div v-for="s in userSessions.slice(0,20)" :key="s.session_id" class="session-item" @click="switchSession(s.session_id)" :class="{active: sessionId===s.session_id}">
            <div class="session-row">
              <span class="session-preview">{{ s.preview || '(新会话)' }}</span>
              <el-button text size="small" type="danger" @click.stop="deleteUserSession(s.session_id)" :icon="Delete" style="flex-shrink:0" />
            </div>
            <div class="session-meta">{{ s.message_count }}条 · {{ s.created_at || '刚刚' }}</div>
          </div>
        </el-card>
      </el-aside>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref, computed, nextTick, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Plus, Promotion, Cpu, ChatDotRound, UserFilled, PictureFilled, Microphone, UploadFilled } from '@element-plus/icons-vue'

// ── 登录 ──
const token = ref(localStorage.getItem('auth_token') || '')
const loginUsername = ref('')
const loginPassword = ref('')
const loginError = ref('')
const loggingIn = ref(false)
const currentUser = ref(JSON.parse(localStorage.getItem('auth_user')||'null') || { user_id:'', display_name:'未登录', role:'worker', roleLabel:'访客' })

const demoPwTip = ref('')

async function loadDemoInfo() {
  try {
    const r = await fetch('/api/v1/auth/me')
    const j = await r.json()
    if (j.data?.tip) demoPwTip.value = j.data.tip
  } catch(e) { /* ignore */ }
}
loadDemoInfo()

const demoUsers = [
  { name:'worker_zhang', role:'产线工人(A)' }, { name:'maintainer_li', role:'维修工(A)' },
  { name:'shift_lead_wang', role:'班组长(A)' }, { name:'director_zhao', role:'车间主任(A)' },
  { name:'engineer_chen', role:'工艺工程师' }, { name:'manager_zhou', role:'厂长' },
  { name:'worker_sun', role:'产线工人(B)' }, { name:'maintainer_huang', role:'维修工(B)' },
  { name:'shift_lead_liu', role:'班组长(B)' },
]
const roleLabels = { worker_zhang:'产线工人(A)',maintainer_li:'维修工(A)',shift_lead_wang:'班组长(A)',director_zhao:'车间主任(A)',engineer_chen:'工艺工程师',manager_zhou:'厂长',worker_sun:'产线工人(B)',maintainer_huang:'维修工(B)',shift_lead_liu:'班组长(B)' }

async function login() {
  if (!loginUsername.value.trim()) { loginError.value='请输入用户名'; return }
  loginError.value = ''
  loggingIn.value = true
  try {
    const r = await fetch('/api/v1/auth/login',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({username:loginUsername.value,password:loginPassword.value}),
    })
    const j = await r.json()
    if (j.code===0) {
      token.value = j.data.token
      localStorage.setItem('auth_token', j.data.token)
      const u = {
        user_id:j.data.user_id,display_name:j.data.display_name,
        role:j.data.role,roleLabel:roleLabels[j.data.user_id]||j.data.role,
      }
      currentUser.value = u
      localStorage.setItem('auth_user', JSON.stringify(u))
      loginPassword.value = ''
      // 加载用户历史会话 + 恢复最后会话
      loadUserSessions()
      if (sessionId.value) restoreSession()
    } else { loginError.value = j.message||'登录失败' }
  } catch(e) { loginError.value = '网络错误: '+e.message }
  finally { loggingIn.value = false }
}

function logout() {
  ElMessageBox.confirm('确定要退出登录吗？', '退出确认', { confirmButtonText:'确定', cancelButtonText:'取消', type:'warning' }).then(() => {
    token.value = ''
    localStorage.removeItem('auth_token')
    localStorage.removeItem('auth_user')
    messages.value = []
    sessionId.value = ''
    userSessions.value = []
    currentUser.value = { user_id:'',display_name:'未登录',role:'worker',roleLabel:'访客' }
  }).catch(() => {})
}

async function loadUserSessions() {
  try {
    const r = await fetch('/api/v1/conversation/sessions', { headers: {'Authorization':`Bearer ${token.value}`} })
    const j = await r.json()
    userSessions.value = j.data?.sessions || []
  } catch(e) { console.error('loadUserSessions:', e) }
}

async function restoreSession(sid) {
  const targetSid = sid || sessionId.value
  if (!targetSid) return
  try {
    const r = await fetch(`/api/v1/conversation/session/${targetSid}/restore`, { headers: {'Authorization':`Bearer ${token.value}`} })
    const j = await r.json()
    if (j.code===0 && j.data.messages.length) {
      messages.value = j.data.messages.map(m => {
        const imgMatch = (m.content||'').match(/^\[IMG\](\S+)\s*/)
        if (imgMatch) {
          return {...m, image: imgMatch[1], content: (m.content||'').replace(imgMatch[0], '').trim() || '[图片]', time: m.time||''}
        }
        return {...m, time: m.time||''}
      })
      sessionId.value = targetSid
      localStorage.setItem('chat_session_id', targetSid)
      scrollToBottom()
    }
  } catch(e) { /* ignore */ }
}

onMounted(() => {
  if (token.value) {
    // 验证 token 是否仍然有效
    fetch('/api/v1/auth/me', { headers: {'Authorization': `Bearer ${token.value}`} })
      .then(r => { if (r.status === 401) { logout(); return } })
      .catch(() => logout())
    loadUserSessions()
    if (sessionId.value) restoreSession()
  }
})

// ── 语音输入 ──
const listening = ref(false)
let recognition = null

function toggleVoice() {
  if (listening.value) { stopVoice(); return }
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    ElMessage.warning('当前浏览器不支持语音输入，请使用 Chrome 或 Edge')
    return
  }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition
  recognition = new SR()
  recognition.lang = 'zh-CN'
  recognition.interimResults = true
  recognition.continuous = false
  recognition.onresult = (e) => {
    let text = ''
    for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript
    inputText.value = text
  }
  recognition.onend = () => { listening.value = false }
  recognition.onerror = (e) => { listening.value = false; if (e.error !== 'no-speech') ElMessage.error('语音识别出错: ' + e.error) }
  recognition.start()
  listening.value = true
}

function stopVoice() { if (recognition) { recognition.stop(); listening.value = false } }

// ── 图片上传 ──
const uploadedImages = ref([])

function previewImage(src) { window.open(src, '_blank') }

async function onImageSelect(file) {
  const reader = new FileReader()
  reader.onload = async (e) => {
    const previewSrc = e.target.result
    messages.value.push({ role:'user', content:'[图片分析中...]', image:previewSrc, time:new Date().toLocaleTimeString() })
    scrollToBottom()
    streaming.value = true; streamingText.value = ''

    try {
      const fd = new FormData()
      fd.append('file', file.raw || file)
      fd.append('message', inputText.value || '请分析这张图片中的设备信息')
      fd.append('workshop_id', currentWorkshop.value)
      fd.append('session_id', sessionId.value || '')
      const headers = {}
      if (token.value) headers['Authorization'] = `Bearer ${token.value}`
      const r = await fetch('/api/v1/conversation/image-chat', { method:'POST', headers, body:fd })
      const j = await r.json()
      if (j.code === 0) {
        // 更新用户消息——用服务器返回的永久URL替换base64预览
        const imageUrl = j.data.image_url
        const userMsg = messages.value[messages.value.length-1]
        userMsg.image = imageUrl
        userMsg.content = inputText.value ? `[图片] ${inputText.value}` : '[图片]'
        // 显示 AI 回答
        const answer = j.data.answer
        messages.value.push({ role:'assistant', content:answer, time:new Date().toLocaleTimeString() })
        sessionId.value = j.data.session_id
        localStorage.setItem('chat_session_id', j.data.session_id)
        loadUserSessions()
      }
    } catch(e) {
      messages.value[messages.value.length-1].content = '[图片] 分析失败: ' + e.message
    } finally {
      streaming.value = false; streamingText.value = ''
      scrollToBottom()
    }
  }
  reader.readAsDataURL(file.raw || file)
}

function newSession() {
  const sid = 'session_' + Date.now().toString(36)
  sessionId.value = sid
  localStorage.setItem('chat_session_id', sid)
  messages.value = []
  // 即时加入本地列表，避免刷新后空会话消失
  userSessions.value.unshift({
    session_id: sid, preview: '(新会话)', message_count: 0,
    last_active: new Date().toLocaleString('zh-CN').replace(/\//g,'-'),
  })
}

function switchSession(sid) {
  if (sid === sessionId.value) return
  sessionId.value = sid
  localStorage.setItem('chat_session_id', sid)
  messages.value = []
  restoreSession(sid)
  loadUserSessions()
}

async function deleteUserSession(sid) {
  try {
    await fetch(`/api/v1/conversation/${sid}`, { method:'DELETE', headers:{'Authorization':`Bearer ${token.value}`} })
    if (sid === sessionId.value) newSession()
    loadUserSessions()
  } catch(e) { /* ignore */ }
}

async function cleanupOld() {
  try {
    await fetch('/api/v1/conversation/cleanup', { method:'POST', headers:{'Authorization':`Bearer ${token.value}`} })
    loadUserSessions()
    ElMessage.success('旧会话已清理')
  } catch(e) { ElMessage.error('清理失败') }
}

// ── 聊天 ──
const currentWorkshop = ref('workshop-a')
const inputText = ref('')
const messages = ref([])
const streaming = ref(false)
const streamingText = ref('')
const sessionId = ref(localStorage.getItem('chat_session_id') || '')
const userSessions = ref([])
const messagesContainer = ref(null)
const equipmentList = ['海天MA1200','FANUC CNC','扬力冲压线','HA-003','HA-005']
const recentCitations = computed(() => {
  const all = []; for (const msg of [...messages.value].reverse()) { if (msg.citations) all.push(...msg.citations); if (all.length>=5) break }; return all.slice(0,5)
})

async function scrollToBottom() { await nextTick(); const el = messagesContainer.value; if (el) el.scrollTop = el.scrollHeight }

async function sendMessage() {
  const text = inputText.value.trim(); if (!text||streaming.value) return
  inputText.value = ''
  messages.value.push({ role:'user', content:text, time:new Date().toLocaleTimeString() })
  await scrollToBottom()
  streaming.value = true; streamingText.value = ''

  try {
    const headers = { 'Content-Type':'application/json' }
    if (token.value) headers['Authorization'] = `Bearer ${token.value}`
    const response = await fetch('/api/v1/conversation/chat',{
      method:'POST',headers,
      body:JSON.stringify({ message:text, workshop_id:currentWorkshop.value, session_id:sessionId.value||null }),
    })
    if (response.status === 401) { logout(); throw new Error('登录已过期，请重新登录') }
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const reader = response.body.getReader(); const decoder = new TextDecoder()
    let buffer = '', assistantContent = '', citations = [], modelUsed = ''
    while (true) {
      const { done, value } = await reader.read(); if (done) break
      buffer += decoder.decode(value,{stream:true}); const lines = buffer.split('\n'); buffer = lines.pop()||''
      for (const line of lines) {
        if (line.startsWith('data:')) {
          try {
            const d = JSON.parse(line.slice(6))
            if (d.type==='text') { assistantContent += d.content; streamingText.value = assistantContent; await scrollToBottom() }
            else if (d.session_id) { sessionId.value = d.session_id; localStorage.setItem('chat_session_id', d.session_id); modelUsed = d.model_used }
            else if (d.chunk_id) citations.push(d)
          } catch {}
        }
      }
    }
    messages.value.push({ role:'assistant',content:assistantContent||'未获取到回答，请检查 DeepSeek API Key 是否已配置。',citations,model:modelUsed,time:new Date().toLocaleTimeString() })
  } catch(e) { messages.value.push({ role:'assistant',content:`请求失败: ${e.message}`,time:new Date().toLocaleTimeString() }) }
  finally { streaming.value = false; streamingText.value = ''; await scrollToBottom() }
}

function quickAction(action) {
  const prompts = { '查SOP':'请帮我查找SOP，关于','查报警码':'请查一下报警码','故障处理':'设备出现故障，','设备点检':'设备每日点检需要检查哪些项目？','换模步骤':'注塑机换模的具体步骤是什么？' }
  inputText.value = prompts[action] || `${action} 相关问题`
}

function formatMessage(text) {
  if (!text) return ''
  return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>')
    .replace(/⚠️/g,'<span style=\"color:#E6A23C\">⚠️</span>')
    .replace(/步骤(\d+):/g,'<br><strong>步骤$1:</strong>').replace(/\n/g,'<br>')
}
</script>

<style>
* { margin:0; padding:0; box-sizing:border-box }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif }
/* 登录 */
.login-page { display:flex; align-items:center; justify-content:center; height:100vh; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%) }
.login-card { background:white; border-radius:12px; padding:40px; width:420px; box-shadow:0 20px 60px rgba(0,0,0,.3) }
.login-card h2 { text-align:center; color:#303133; margin-bottom:4px }
.login-subtitle { text-align:center; color:#909399; font-size:14px; margin-bottom:24px }
.login-hint { margin-top:16px; font-size:13px }
.login-hint p { margin-bottom:8px; color:#606266 }
/* 聊天 */
.app-container { height:100vh; display:flex; flex-direction:column }
.app-header { display:flex; align-items:center; justify-content:space-between; background:#409EFF; color:white; padding:0 20px; height:56px; flex-shrink:0 }
.app-header h2 { font-size:18px; font-weight:600; white-space:nowrap }
.header-right { display:flex; align-items:center; flex-shrink:0; gap:4px }
.main-area { flex:1; overflow:hidden }
.chat-area { display:flex; flex-direction:column; background:#f5f7fa; padding:0 }
.messages { flex:1; overflow-y:auto; padding:20px }
.welcome { text-align:center; padding:60px 20px; color:#909399 }
.welcome p { margin-top:16px; font-size:16px }
.message { display:flex; margin-bottom:20px }
.message.user { flex-direction:row-reverse }
.message-avatar { flex-shrink:0 }
.message.user .message-avatar { margin-left:10px }
.message.assistant .message-avatar { margin-right:10px }
.message-content { max-width:70% }
.message.user .message-content { text-align:right }
.message-text { background:white; border-radius:12px; padding:12px 16px; line-height:1.6; font-size:14px; box-shadow:0 1px 3px rgba(0,0,0,.08) }
.message.user .message-text { background:#409EFF; color:white }
.message-meta { font-size:11px; color:#909399; margin-top:4px }
.thinking-text { color:#909399; font-size:14px; padding:8px 0; display:flex; align-items:center }
.dot-anim { display:inline-block; width:18px; animation:dotPulse 1.4s infinite }
@keyframes dotPulse { 0%,20%{opacity:0} 50%{opacity:1} 80%,100%{opacity:0} }
.citations { text-align:left }
.input-area { padding:12px 20px 16px; background:white; border-top:1px solid #e4e7ed }
.quick-actions { margin-bottom:10px; display:flex; gap:6px; flex-wrap:wrap }
.input-row { display:flex; align-items:center }
.info-panel { background:white; border-left:1px solid #e4e7ed; padding:16px; overflow-y:auto }
.info-panel .el-card { margin-bottom:12px }
.text-secondary { color:#909399; font-size:13px }
.session-item { padding:6px 8px; margin:4px 0; border-radius:6px; cursor:pointer; border:1px solid #eee }
.session-item:hover { background:#ecf5ff }
.session-item.active { background:#ecf5ff; border-color:#409EFF }
.session-row { display:flex; align-items:center; justify-content:space-between }
.session-preview { font-size:12px; color:#303133; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; min-width:0 }
.session-meta { font-size:11px; color:#909399; margin-top:2px }
@media (max-width:768px) {
  .info-panel { display:none }
  .message-content { max-width:85% }
  .app-header { padding:0 10px }
  .app-header h2 { font-size:14px }
  /* 隐藏角色标签，只保留用户名 */
  .header-right .el-tag--info { display:none }
  .header-right .el-select { width:90px !important }
}
@media (max-width:480px) {
  .app-header h2 { font-size:13px }
  .header-right { gap:2px }
  .header-right .el-button { padding:0 4px; font-size:12px }
}
</style>
