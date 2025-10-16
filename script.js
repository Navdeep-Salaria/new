// Initial mock appliances
let mockAppliances = [
  { id:'a1', name:'Tube Light', watt:15, count:2, enabled:true },
  { id:'a2', name:'LED Bulb', watt:9, count:4, enabled:true },
  { id:'a3', name:'Fan', watt:65, count:2, enabled:true },
];

// State
let isRunning = false;
let totalKwh = 0;
let simSeconds = 0;
let tickInterval = null;
const TICK_MS = 1000;

// Shortcuts
const el = id => document.getElementById(id);

// Format time hh:mm:ss
function formatTime(sec){
  const h = String(Math.floor(sec/3600)).padStart(2,'0');
  const m = String(Math.floor((sec%3600)/60)).padStart(2,'0');
  const s = String(sec%60).padStart(2,'0');
  return `${h}:${m}:${s}`;
}

// Compute instantaneous power
function computePower() {
  return mockAppliances
    .filter(a=>a.enabled)
    .reduce((sum,a)=> sum + a.watt * a.count, 0);
}

// Tick function
function tick(speed=1) {
  simSeconds += 1 * speed;

  const watts = computePower();
  const kwhAdd = (watts/1000) * (1/3600) * speed;
  totalKwh += kwhAdd;

  el('displayPower').textContent = `${watts} W`;
  el('displayKwh').textContent = `${totalKwh.toFixed(6)} kWh`;
  el('simTime').textContent = formatTime(simSeconds);
}

// Render appliances
function renderAppliances(){
  const list = el('applianceList');
  list.innerHTML = '';
  mockAppliances.forEach(a=>{
    const row = document.createElement('div');
    row.className = 'appliance';
    row.innerHTML = `
      <div>
        <strong>${a.name}</strong><br>
        <span class="muted">${a.watt}W × ${a.count}</span>
      </div>
      <button onclick="toggleAppliance('${a.id}')">
        ${a.enabled ? 'On' : 'Off'}
      </button>
    `;
    list.appendChild(row);
  });
  el('applianceCount').textContent = mockAppliances.length;
}

// Toggle appliance on/off
function toggleAppliance(id){
  const app = mockAppliances.find(x=>x.id===id);
  if(app) { app.enabled = !app.enabled; renderAppliances(); }
}

// Controls
el('startBtn').onclick = ()=>{
  if(isRunning) return;
  isRunning = true;
  tickInterval = setInterval(()=>tick(Number(el('speedSelect').value)), TICK_MS);
  el('startBtn').disabled = true;
  el('pauseBtn').disabled = false;
};

el('pauseBtn').onclick = ()=>{
  isRunning = false;
  clearInterval(tickInterval);
  el('startBtn').disabled = false;
  el('pauseBtn').disabled = true;
};

el('resetBtn').onclick = ()=>{
  isRunning = false;
  clearInterval(tickInterval);
  totalKwh=0; simSeconds=0;
  el('displayPower').textContent = '0 W';
  el('displayKwh').textContent = '0.000 kWh';
  el('simTime').textContent = '00:00:00';
  el('startBtn').disabled = false;
  el('pauseBtn').disabled = true;
};

// Add appliance
el('addApplianceBtn').onclick = ()=>{
  const name = el('newName').value || 'New Appliance';
  const watt = Number(el('newWatt').value) || 10;
  const count = Number(el('newCount').value) || 1;
  mockAppliances.push({ id:'id'+Date.now(), name, watt, count, enabled:true });
  renderAppliances();
  el('newName').value = '';
  el('newWatt').value = '';
  el('newCount').value = 1;
};

// Demo scenario
el('loadScenarioBtn').onclick = ()=>{
  mockAppliances = [
    { id:'b1', name:'Fridge', watt:120, count:1, enabled:true },
    { id:'b2', name:'AC', watt:900, count:1, enabled:false },
    { id:'b3', name:'Bulbs', watt:10, count:5, enabled:true },
  ];
  renderAppliances();
};

// Init
renderAppliances();
