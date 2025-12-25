/**
 * SemanticBridge - Frontend Application
 * Wizard-based UI for connecting people through unexpected stories
 */

const API_URL = 'http://localhost:8000';

// App State
const state = {
    currentStep: 1,
    context: null,      // 'team', 'strangers', 'couples'
    mode: null,         // 'max_distance', 'surprise_bridge', 'asymmetric_gift'
    participantCount: 2,
    participants: [],   // Array of { id, name, bites: [] }
    temperature: 0.8,
    stream: true,
    results: null
};

// DOM Elements
const wizard = document.querySelector('.wizard');
const steps = document.querySelectorAll('.step');
const loadingOverlay = document.querySelector('.loading-overlay');
const loadingText = document.querySelector('.loading-text');
let loadingTimers = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initContextCards();
    initModeCards();
    initParticipantCounter();
    initNavigation();
    initSettings();
});

// === Step 1: Context Selection ===
function initContextCards() {
    const cards = document.querySelectorAll('.context-card');
    cards.forEach(card => {
        card.addEventListener('click', () => {
            // Remove previous selection
            cards.forEach(c => c.classList.remove('selected'));
            // Select this one
            card.classList.add('selected');
            state.context = card.dataset.context;
            
            // Auto-advance after short delay
            setTimeout(() => goToStep(2), 300);
        });
    });
}

// === Step 2: Mode Selection ===
function initModeCards() {
    const cards = document.querySelectorAll('.mode-card');
    
    cards.forEach(card => {
        card.addEventListener('click', () => {
            if (card.disabled) return;
            
            cards.forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');
            state.mode = card.dataset.mode;

            if (state.mode === 'triplet_weave' && state.participantCount < 3) {
                state.participantCount = 3;
                document.getElementById('participant-count').value = 3;
            }
            
            updateStep2NextButton();
        });
    });
}

function updateModeCardsForContext() {
    const asymmetricCard = document.querySelector('.mode-card[data-mode="asymmetric_gift"]');
    const tripletCard = document.querySelector('.mode-card[data-mode="triplet_weave"]');
    const centroidCard = document.querySelector('.mode-card[data-mode="centroid_constellation"]');
    const chainCard = document.querySelector('.mode-card[data-mode="bridge_chain"]');
    
    // Asymmetric gift only makes sense for couples
    if (state.context === 'couples') {
        asymmetricCard.disabled = false;
        asymmetricCard.style.opacity = '1';
    } else {
        asymmetricCard.disabled = true;
        asymmetricCard.style.opacity = '0.4';
        
        // If it was selected, deselect
        if (state.mode === 'asymmetric_gift') {
            asymmetricCard.classList.remove('selected');
            state.mode = null;
        }
    }

    // Triplet weave requires 3 participants
    if (state.context === 'couples') {
        tripletCard.disabled = true;
        tripletCard.style.opacity = '0.4';
        if (state.mode === 'triplet_weave') {
            tripletCard.classList.remove('selected');
            state.mode = null;
        }
    } else {
        tripletCard.disabled = false;
        tripletCard.style.opacity = '1';
    }

    // Group modes are allowed for all contexts, but couples should add more bites
    if (state.context === 'couples') {
        centroidCard.disabled = false;
        centroidCard.style.opacity = '1';
        chainCard.disabled = false;
        chainCard.style.opacity = '1';
    }
    
    // For couples, default participant count to 2
    if (state.context === 'couples') {
        state.participantCount = 2;
        document.getElementById('participant-count').value = 2;
    }
}

function updateStep2NextButton() {
    const nextBtn = document.querySelector('.step-2 .next-btn');
    nextBtn.disabled = !state.mode;
}

// === Participant Counter ===
function initParticipantCounter() {
    const input = document.getElementById('participant-count');
    const minusBtn = document.querySelector('.counter-btn.minus');
    const plusBtn = document.querySelector('.counter-btn.plus');
    
    minusBtn.addEventListener('click', () => {
        if (state.participantCount > 2) {
            state.participantCount--;
            input.value = state.participantCount;
        }
    });
    
    plusBtn.addEventListener('click', () => {
        if (state.participantCount < 8) {
            state.participantCount++;
            input.value = state.participantCount;
        }
    });
    
    input.addEventListener('change', () => {
        let val = parseInt(input.value);
        if (isNaN(val) || val < 2) val = 2;
        if (val > 8) val = 8;
        state.participantCount = val;
        input.value = val;
    });
}

// === Navigation ===
function initNavigation() {
    // Back buttons
    document.querySelectorAll('.back-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            goToStep(state.currentStep - 1);
        });
    });
    
    // Step 2 next button
    document.querySelector('.step-2 .next-btn').addEventListener('click', () => {
        initializeParticipants();
        goToStep(3);
    });
    
    // Generate button
    document.querySelector('.generate-btn').addEventListener('click', generate);
    
    // Restart button
    document.querySelector('.restart-btn').addEventListener('click', () => {
        resetState();
        goToStep(1);
    });
}

function goToStep(stepNum) {
    state.currentStep = stepNum;
    
    steps.forEach(step => {
        step.classList.remove('active');
        if (parseInt(step.dataset.step) === stepNum) {
            step.classList.add('active');
        }
    });
    
    // Step-specific setup
    if (stepNum === 2) {
        updateModeCardsForContext();
        updateStep2NextButton();
    }
}

// === Step 3: Participant Input ===
function initializeParticipants() {
    state.participants = [];
    
    for (let i = 0; i < state.participantCount; i++) {
        state.participants.push({
            id: `p${i + 1}`,
            name: `Person ${i + 1}`,
            bites: ['']
        });
    }
    
    renderParticipants();
}

function renderParticipants() {
    const container = document.querySelector('.participants-container');
    container.innerHTML = '';
    
    state.participants.forEach((participant, pIndex) => {
        const card = document.createElement('div');
        card.className = 'participant-card';
        card.dataset.participantIndex = pIndex;
        
        const placeholders = getPlaceholders(pIndex);
        
        card.innerHTML = `
            <div class="participant-header">
                <div class="participant-number">${pIndex + 1}</div>
                <input type="text" 
                       class="participant-name-input" 
                       placeholder="Name" 
                       value="${participant.name}"
                       data-participant="${pIndex}">
            </div>
            <div class="bites-container">
                ${participant.bites.map((bite, bIndex) => `
                    <div class="bite-input-wrapper" data-bite-index="${bIndex}">
                        <textarea class="bite-input" 
                                  placeholder="${placeholders[bIndex % placeholders.length]}"
                                  rows="2"
                                  data-participant="${pIndex}"
                                  data-bite="${bIndex}">${bite}</textarea>
                        ${participant.bites.length > 1 ? `
                            <button class="remove-bite-btn" data-participant="${pIndex}" data-bite="${bIndex}">×</button>
                        ` : ''}
                    </div>
                `).join('')}
                <button class="add-bite-btn" data-participant="${pIndex}">+ Add another knowledge bite</button>
            </div>
        `;
        
        container.appendChild(card);
    });
    
    // Add event listeners
    container.querySelectorAll('.participant-name-input').forEach(input => {
        input.addEventListener('input', (e) => {
            const pIndex = parseInt(e.target.dataset.participant);
            state.participants[pIndex].name = e.target.value || `Person ${pIndex + 1}`;
        });
    });
    
    container.querySelectorAll('.bite-input').forEach(textarea => {
        textarea.addEventListener('input', (e) => {
            const pIndex = parseInt(e.target.dataset.participant);
            const bIndex = parseInt(e.target.dataset.bite);
            state.participants[pIndex].bites[bIndex] = e.target.value;
        });
    });
    
    container.querySelectorAll('.add-bite-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const pIndex = parseInt(e.target.dataset.participant);
            state.participants[pIndex].bites.push('');
            renderParticipants();
        });
    });
    
    container.querySelectorAll('.remove-bite-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const pIndex = parseInt(e.target.dataset.participant);
            const bIndex = parseInt(e.target.dataset.bite);
            state.participants[pIndex].bites.splice(bIndex, 1);
            renderParticipants();
        });
    });
}

function getPlaceholders(participantIndex) {
    const allPlaceholders = [
        ["A memory from childhood...", "Something you're obsessed with...", "A random fact you love...", "A skill nobody knows you have..."],
        ["Your strangest hobby...", "A place that changed you...", "Something you've always wanted to try...", "A fear you've overcome..."],
        ["Your comfort food memory...", "A book that rewired your brain...", "Your hidden talent...", "A dream you keep having..."],
        ["The smell that takes you back...", "A skill you wish you had...", "Your guilty pleasure...", "A moment of unexpected joy..."]
    ];
    return allPlaceholders[participantIndex % allPlaceholders.length];
}

// === Generation ===
async function generate() {
    // Validate inputs
    const validParticipants = state.participants.filter(p => 
        p.bites.some(b => b.trim().length > 0)
    );
    
    if (validParticipants.length < 2) {
        alert('Please enter at least one knowledge bite for at least 2 participants.');
        return;
    }

    if (state.mode === 'triplet_weave' && validParticipants.length < 3) {
        alert('Triplet Weave requires at least 3 participants with knowledge bites.');
        return;
    }

    if (state.context === 'couples' && ['centroid_constellation', 'bridge_chain'].includes(state.mode)) {
        const biteCounts = state.participants.map(p => p.bites.filter(b => b.trim().length > 0).length);
        if (biteCounts.some(count => count < 2)) {
            alert('For couples, group modes require at least 2 knowledge bites per person.');
            return;
        }
    }
    
    // Prepare request
    const requestData = {
        context: state.context,
        mode: state.mode,
        participants: state.participants.map(p => ({
            id: p.id,
            name: p.name || `Person ${state.participants.indexOf(p) + 1}`,
            bites: p.bites.filter(b => b.trim().length > 0)
        })).filter(p => p.bites.length > 0),
        temperature: state.temperature,
        stream: false  // Disabled for now - state.stream
    };
    
    showLoading('Embedding your knowledge bites...');
    
    try {
        if (state.stream) {
            await generateWithStream(requestData);
        } else {
            await generateWithoutStream(requestData);
        }
    } catch (error) {
        console.error('Generation error:', error);
        hideLoading();
        alert('Error generating connections: ' + error.message);
    }
}

async function generateWithoutStream(requestData) {
    updateLoadingText('Finding semantic bridges...');
    scheduleLoadingHints();
    
    const response = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Generation failed');
    }
    
    const data = await response.json();
    state.results = data;
    
    hideLoading();
    renderResults(data.connections, data.story, data.debug, data.reasoning, data.thinking, data.groups);
    goToStep(4);
}

async function generateWithStream(requestData) {
    updateLoadingText('Finding semantic bridges...');
    scheduleLoadingHints();
    
    const response = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Generation failed');
    }
    
    // For streaming, we need to handle the response differently
    // The current backend returns streaming text, but connections come first
    // Let's read the full response for now (we can improve streaming later)
    const text = await response.text();
    
    // Parse as JSON if possible (non-streaming fallback)
    try {
        const data = JSON.parse(text);
        state.results = data;
        hideLoading();
        renderResults(data.connections, data.story, data.debug, data.reasoning, data.thinking, data.groups);
        goToStep(4);
    } catch {
        // If it's streamed text, show it directly
        hideLoading();
        renderResults([], text, null, null, null, []);
        goToStep(4);
    }
}

function renderResults(connections, story, debug, reasoning, thinking, groups) {
    const connectionsSection = document.querySelector('.connections-found');
    const connectionsList = document.querySelector('.connections-list');
    const groupSection = document.querySelector('.group-connections-found');
    const groupList = document.querySelector('.group-connections-list');
    const storyContainer = document.querySelector('.story-content');
    const embeddingDetails = document.getElementById('embedding-insights');
    const embeddingText = document.getElementById('embedding-insights-text');
    const reasoningDetails = document.getElementById('reasoning-insights');
    const reasoningText = document.getElementById('reasoning-text');
    const reasoningSummary = reasoningDetails ? reasoningDetails.querySelector('summary') : null;
    
    // Render connections
    if (connectionsList && connectionsSection) {
        connectionsList.innerHTML = connections.map((conn, i) => `
            <div class="connection-card">
                <div class="connection-header">
                    <span class="connection-label">Connection ${i + 1}</span>
                    <span class="connection-distance">Distance: ${conn.distance.toFixed(3)}</span>
                </div>
                <div class="connection-bites">
                    <div class="connection-bite">
                        <span class="bite-author">${conn.participant1}:</span>
                        <span class="bite-text">"${conn.bite1}"</span>
                    </div>
                    <div class="connection-bite">
                        <span class="bite-author">${conn.participant2}:</span>
                        <span class="bite-text">"${conn.bite2}"</span>
                    </div>
                </div>
                ${conn.bridge_concept ? `
                    <div class="bridge-concept">
                        <strong>Bridge:</strong> ${conn.bridge_concept}
                    </div>
                ` : ''}
            </div>
        `).join('');
        connectionsSection.style.display = connections.length ? 'flex' : 'none';
    }

    if (groupList && groupSection) {
        groupList.innerHTML = (groups || []).map((group, i) => `
            <div class="connection-card">
                <div class="connection-header">
                    <span class="connection-label">Group ${i + 1}</span>
                    <span class="connection-distance">Score: ${group.score.toFixed(3)}</span>
                </div>
                <div class="connection-bites">
                    ${group.members.map(member => `
                        <div class="connection-bite">
                            <span class="bite-author">${member.participant}:</span>
                            <span class="bite-text">"${member.bite}"</span>
                        </div>
                    `).join('')}
                </div>
                <div class="bridge-concept">
                    <strong>Strategy:</strong> ${formatStrategyLabel(group.strategy)}
                </div>
            </div>
        `).join('');
        groupSection.style.display = groups && groups.length ? 'flex' : 'none';
    }
    
    // Render story with typewriter effect
    storyContainer.innerHTML = '';
    typewriterEffect(storyContainer, story || '');

    // Render embedding/debug insights
    if (debug && embeddingText && embeddingDetails) {
        embeddingText.textContent = formatEmbeddingDebug(debug);
        embeddingDetails.style.display = 'block';
    } else if (embeddingDetails) {
        embeddingDetails.style.display = 'none';
    }

    // Render reasoning if provided (collapsed by default)
    if (reasoningText && reasoningDetails) {
        const thinkingText = thinking && thinking.trim().length > 0 ? thinking.trim() : '';
        const reasoningTextValue = reasoning && reasoning.trim().length > 0 ? reasoning.trim() : '';
        const finalText = thinkingText || reasoningTextValue;
        if (finalText) {
            reasoningText.textContent = finalText;
            if (reasoningSummary) {
                reasoningSummary.textContent = thinkingText ? 'Model thinking (captured)' : 'Model reasoning (collapsed)';
            }
            reasoningDetails.style.display = 'block';
        } else {
            reasoningDetails.style.display = 'none';
        }
    }
}

function typewriterEffect(element, text, speed = 15) {
    let i = 0;
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    element.textContent = '';
    element.appendChild(cursor);
    
    function type() {
        if (i < text.length) {
            element.textContent = text.substring(0, i + 1);
            element.appendChild(cursor);
            i++;
            setTimeout(type, speed);
        } else {
            element.textContent = text;
        }
    }
    
    type();
}

function formatEmbeddingDebug(debug) {
    const lines = [];
    const search = debug.search || {};
    const embedding = debug.embedding || {};
    const counts = search.pair_counts || {};
    const threshold = search.threshold || {};
    const statsAll = (search.distance_stats || {}).all_pairs || {};
    const statsInRange = (search.distance_stats || {}).within_threshold || {};

    lines.push(`Context: ${debug.context || 'n/a'}`);
    lines.push(`Mode: requested=${debug.mode_requested || 'n/a'}, used=${debug.mode_used || 'n/a'}`);
    if (debug.mode_note) {
        lines.push(`Note: ${debug.mode_note}`);
    }
    lines.push(`Participants: ${debug.participant_count || 0}`);
    lines.push(`Knowledge bites: ${debug.bite_count || 0}`);
    if (debug.group_count) {
        lines.push(`Group connections: ${debug.group_count}`);
    }
    if (embedding.model || embedding.dimensions) {
        lines.push(`Embedding model: ${embedding.model || 'n/a'} (${embedding.dimensions || 0} dims)`);
    }
    lines.push(`Pairing strategy: ${debug.pairing_strategy || 'pairwise'}`);
    if (search.strategy) {
        lines.push(`Search strategy: ${search.strategy}`);
    }
    if (threshold.min !== undefined && threshold.max !== undefined) {
        lines.push(`Distance threshold: ${threshold.min} - ${threshold.max}`);
    }
    if (Object.keys(counts).length > 0) {
        const countParts = [];
        if ('total_pairs' in counts) countParts.push(`total=${counts.total_pairs}`);
        if ('cross_participant_pairs' in counts) countParts.push(`cross=${counts.cross_participant_pairs}`);
        if ('with_embeddings' in counts) countParts.push(`with_embeddings=${counts.with_embeddings}`);
        if ('within_threshold' in counts) countParts.push(`in_range=${counts.within_threshold}`);
        if (countParts.length > 0) {
            lines.push(`Pairs: ${countParts.join(', ')}`);
        }
    }
    if (Object.keys(statsAll).length > 0) {
        lines.push(`Distance stats (all): min=${formatNumber(statsAll.min)}, max=${formatNumber(statsAll.max)}, mean=${formatNumber(statsAll.mean)}`);
    }
    if (Object.keys(statsInRange).length > 0) {
        lines.push(`Distance stats (in range): min=${formatNumber(statsInRange.min)}, max=${formatNumber(statsInRange.max)}, mean=${formatNumber(statsInRange.mean)}`);
    }
    if (search.score_stats) {
        lines.push(`Group score stats: min=${formatNumber(search.score_stats.min)}, max=${formatNumber(search.score_stats.max)}, mean=${formatNumber(search.score_stats.mean)}`);
    }
    if (Array.isArray(search.candidate_pairs) && search.candidate_pairs.length > 0) {
        lines.push('Top candidate pairs:');
        search.candidate_pairs.forEach((pair, idx) => {
            const bite1 = truncateText(pair.bite1 || '', 80);
            const bite2 = truncateText(pair.bite2 || '', 80);
            lines.push(`${idx + 1}. ${formatNumber(pair.distance)} | ${pair.participant1}: "${bite1}" <> ${pair.participant2}: "${bite2}"`);
        });
    }
    if (Array.isArray(search.candidate_groups) && search.candidate_groups.length > 0) {
        lines.push('Top candidate groups:');
        search.candidate_groups.forEach((group, idx) => {
            const members = group.members
                .map(member => `${member.participant}: "${truncateText(member.bite || '', 60)}"`)
                .join(' | ');
            lines.push(`${idx + 1}. ${formatNumber(group.score)} | ${members}`);
        });
    }
    if (search.triplet_counts) {
        lines.push(`Triplets: total=${search.triplet_counts.total_triplets || 0}, in_range=${search.triplet_counts.within_threshold || 0}`);
    }
    if (search.centroid_distance_stats) {
        lines.push(`Centroid distance stats: min=${formatNumber(search.centroid_distance_stats.min)}, max=${formatNumber(search.centroid_distance_stats.max)}, mean=${formatNumber(search.centroid_distance_stats.mean)}`);
    }
    if (Array.isArray(search.steps) && search.steps.length > 0) {
        lines.push('Bridge chain:');
        search.steps.forEach((step, idx) => {
            lines.push(`${idx + 1}. ${step.from} -> ${step.to} (${formatNumber(step.distance)})`);
        });
    }

    return lines.join('\n');
}

function truncateText(text, maxLength) {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength - 3) + '...';
}

function formatNumber(value) {
    if (typeof value !== 'number') return '0.000';
    return value.toFixed(3);
}

function formatStrategyLabel(strategy) {
    const labels = {
        triplet_weave: 'Triplet Weave',
        centroid_constellation: 'Centroid Constellation',
        bridge_chain: 'Bridge Chain'
    };
    return labels[strategy] || strategy || 'Group';
}

// === Settings ===
function initSettings() {
    const panel = document.querySelector('.settings-panel');
    const toggle = document.querySelector('.settings-toggle');
    const tempSlider = document.getElementById('temperature');
    const tempValue = document.querySelector('.temperature-value');
    const streamToggle = document.getElementById('stream-toggle');
    
    toggle.addEventListener('click', () => {
        panel.classList.toggle('open');
    });
    
    tempSlider.addEventListener('input', (e) => {
        state.temperature = parseFloat(e.target.value);
        tempValue.textContent = state.temperature.toFixed(1);
    });
    
    streamToggle.addEventListener('change', (e) => {
        state.stream = e.target.checked;
    });
    
    // Close settings when clicking outside
    document.addEventListener('click', (e) => {
        if (!panel.contains(e.target)) {
            panel.classList.remove('open');
        }
    });
}

// === Utilities ===
function showLoading(text = 'Loading...') {
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
}

function updateLoadingText(text) {
    loadingText.textContent = text;
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
    clearLoadingHints();
}

function resetState() {
    state.currentStep = 1;
    state.context = null;
    state.mode = null;
    state.participantCount = 2;
    state.participants = [];
    state.results = null;
    
    // Reset UI
    document.querySelectorAll('.context-card').forEach(c => c.classList.remove('selected'));
    document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('selected'));
    document.getElementById('participant-count').value = 2;
}

function scheduleLoadingHints() {
    clearLoadingHints();
    loadingTimers.push(setTimeout(() => {
        updateLoadingText('Model thinking...');
    }, 1200));
    loadingTimers.push(setTimeout(() => {
        updateLoadingText('Shaping the story...');
    }, 2600));
}

function clearLoadingHints() {
    loadingTimers.forEach(timer => clearTimeout(timer));
    loadingTimers = [];
}
