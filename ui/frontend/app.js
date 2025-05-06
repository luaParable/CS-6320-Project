const $ = id => document.getElementById(id);
const fileInp = $('fileInp');
const runBtn  = $('btnRun');
const progBar = $('bar');
const outSec  = $('out');
const txtBox  = $('txt');
const entTbl  = $('ent').firstElementChild;
const player  = $('player');
const title   = $('title');
const copyBtn = $('btnCopy');
const saveBtn = $('btnSave');

async function postFile() {
    if (!fileInp.files.length) { alert('Pick a file first'); return null; }
    const fd = new FormData(); fd.append('file', fileInp.files[0]);
    const r = await fetch('/transcribe?fmt=json', { method:'POST', body:fd });
    if (!r.ok){ alert(await r.text()); return null; }
    return r.json();
}

function paintEntities(ents){
    entTbl.innerHTML = "";
    if (!ents.length){
        entTbl.innerHTML = "<tr><td colspan=2><em>none</em></td></tr>"; return;
    }
    ents.forEach(([txt,lab])=>{
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${txt}</td><td class="${lab}">${lab}</td>`;
        entTbl.appendChild(tr);
    });
}

runBtn.onclick = async ()=>{
    if(!fileInp.files.length){alert('choose a file');return;}
    const fd = new FormData(); fd.append('file', fileInp.files[0]);

    progBar.hidden=false; progBar.value=0;
    const r = await fetch('/transcribe',{method:'POST',body:fd});
    progBar.hidden=true;
    if(!r.ok){alert(await r.text());return;}
    const data = await r.json();

    title.textContent = fileInp.files[0].name;
    txtBox.textContent = data.text;                     // ← now shows
    outSec.hidden=false;
    player.src = URL.createObjectURL(fileInp.files[0]);

    // entities
    entTbl.innerHTML="";
    if(!data.entities.length){entTbl.innerHTML="<tr><td colspan=2><em>none</em></td></tr>";}
    data.entities.forEach(([t,l])=>{
        const tr=document.createElement('tr');
        tr.innerHTML=`<td>${t}</td><td class="${l}">${l}</td>`;
        entTbl.appendChild(tr);
    });
};