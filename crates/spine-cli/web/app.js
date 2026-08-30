const fragment=location.hash.slice(1);if(fragment){sessionStorage.setItem('spine-token',fragment);history.replaceState(null,'',location.pathname)}
const token=sessionStorage.getItem('spine-token')||'';
const headers={'content-type':'application/json','x-spine-token':token};
let previous='';
async function api(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers,body:body?JSON.stringify(body):undefined});if(!r.ok)throw new Error(await r.text());return r.status===204?null:r.json().catch(()=>null)}
function esc(text){const d=document.createElement('div');d.textContent=text;return d.innerHTML}
function render(s){const raw=JSON.stringify(s);if(raw===previous)return;previous=raw;
 document.getElementById('phase').textContent=s.phase[0].toUpperCase()+s.phase.slice(1);document.getElementById('activity').textContent=s.activity;document.getElementById('count').textContent=`${s.completed_tools} tool${s.completed_tools===1?'':'s'}`;
 document.getElementById('heart').innerHTML=`<b>${s.incognito?'Incognito heart':'Encrypted heart'}</b><br>${esc(s.heart)}<br>${s.grounding?'Grounding enabled':'Grounding disabled'} · ${s.tool_count} tools`;
 const tools=document.getElementById('tools');tools.innerHTML=s.tools.slice(-30).reverse().map(t=>`<div class="tool ${t.status}"><b>${esc(t.name)}</b><small>${esc(t.status)}</small>${t.input||t.output?`<details><summary>Details</summary>${t.input?`<b>Input</b><pre>${esc(t.input)}</pre>`:''}${t.output?`<b>Output</b><pre>${esc(t.output)}</pre>`:''}</details>`:''}</div>`).join('')||'<small>No tool activity yet</small>';
 const msgs=document.getElementById('messages');if(s.messages.length){msgs.innerHTML=s.messages.map(m=>`<div class="message ${m.role}"><div class="role">${esc(m.role)}</div>${esc(m.content)}</div>`).join('');msgs.scrollTop=msgs.scrollHeight}
 const onboarding=s.phase==='onboarding',input=document.getElementById('input'),send=document.getElementById('send');document.getElementById('notice').textContent=s.notice||'';input.placeholder=onboarding?(s.busy?'Spine is thinking…':'Reply naturally, or /skip'):(s.busy?'Guide the active run at its next tool boundary':'Message Spine');send.textContent=onboarding?'Reply':(s.busy?'Guide':'Send');input.disabled=onboarding&&s.busy;send.disabled=onboarding&&s.busy;
 document.querySelector('[data-action="stop"]').disabled=!s.busy;document.querySelector('[data-action="interrupt"]').disabled=!s.busy;document.querySelector('[data-action="tasks"]').disabled=s.busy;
 document.querySelector('[data-action="resume"]').disabled=!s.checkpoint_available||s.busy;
}
async function poll(){try{render(await api('/api/state'))}catch(e){document.getElementById('notice').textContent=e.message}setTimeout(poll,500)}
document.getElementById('composer').addEventListener('submit',async e=>{e.preventDefault();const input=document.getElementById('input'),message=input.value.trim();if(!message)return;try{const s=await api('/api/state');await api(s.phase==='onboarding'?'/api/message':(s.busy?'/api/guidance':'/api/message'),{message});input.value=''}catch(err){document.getElementById('notice').textContent=err.message}});
document.querySelectorAll('[data-action]').forEach(b=>b.addEventListener('click',()=>api('/api/control',{action:b.dataset.action}).catch(e=>document.getElementById('notice').textContent=e.message)));
document.getElementById('input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('composer').requestSubmit()}});poll();
