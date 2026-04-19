/* ============================================================
   INSIGNIA — AI Interview Preparation Suite
   Application Logic
   ============================================================ */

// ==================== STATE ====================
const AppState = {
    profile: JSON.parse(localStorage.getItem('insignia_profile') || 'null'),
    stats: JSON.parse(localStorage.getItem('insignia_stats') || '{"resumes":0,"questions":0,"topics":0,"mocks":0}'),
    mockSession: null,
    mockTimer: null,
    mockSeconds: 0,
    studyProgress: JSON.parse(localStorage.getItem('insignia_study_progress') || '{}'),
    qaQuestions: [],    // store current Q&A so evaluateAnswer can access question text
};

// ==================== BACKEND CONFIG ====================
const BACKEND_URL = 'http://localhost:8000';

/**
 * Call the Python FastAPI backend.
 * Returns parsed JSON on success, or null on failure (falls back to local logic).
 */
async function callBackend(endpoint, body) {
    try {
        const res = await fetch(`${BACKEND_URL}${endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (err) {
        console.warn(`Backend unavailable (${endpoint}): ${err.message}. Using local fallback.`);
        return null;
    }
}

// ==================== INIT ====================
window.addEventListener('DOMContentLoaded', async () => {
    // Check auth state
    await checkAuthState();

    setTimeout(() => {
        document.getElementById('loading-screen').classList.add('fade-out');
    }, 2000);

    setupNavigation();
    loadProfile();
    updateStats();
    updateReadiness();
    setupOTPInputs();
});

// ==================== NAVIGATION ====================
function setupNavigation() {
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navigateTo(link.dataset.page);
        });
    });
}

function navigateTo(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    const navLink = document.querySelector(`[data-page="${page}"]`);
    if (navLink) navLink.classList.add('active');
}

// ==================== TOAST ====================
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => { toast.classList.add('fade-out'); setTimeout(() => toast.remove(), 300); }, 3000);
}

// ==================== PROFILE ====================
function showProfileModal() { document.getElementById('profile-modal').style.display = 'flex'; }
function closeProfileModal() { document.getElementById('profile-modal').style.display = 'none'; }

function saveProfile(e) {
    e.preventDefault();
    AppState.profile = {
        name: document.getElementById('profile-name').value,
        role: document.getElementById('profile-role').value,
        level: document.getElementById('profile-level').value,
        domain: document.getElementById('profile-domain').value
    };
    localStorage.setItem('insignia_profile', JSON.stringify(AppState.profile));
    loadProfile();
    closeProfileModal();
    showToast('Profile saved successfully!', 'success');
}

function loadProfile() {
    if (AppState.profile) {
        document.getElementById('display-user-name').textContent = AppState.profile.name;
        document.getElementById('display-user-role').textContent = AppState.profile.role || 'No role set';
        document.getElementById('profile-name').value = AppState.profile.name || '';
        document.getElementById('profile-role').value = AppState.profile.role || '';
        document.getElementById('profile-level').value = AppState.profile.level || 'fresher';
        document.getElementById('profile-domain').value = AppState.profile.domain || 'technology';
    }
}

// ==================== STATS ====================
function updateStats() {
    document.getElementById('stat-resumes').textContent = AppState.stats.resumes;
    document.getElementById('stat-questions').textContent = AppState.stats.questions;
    document.getElementById('stat-topics').textContent = AppState.stats.topics;
    document.getElementById('stat-mocks').textContent = AppState.stats.mocks;
}

function incrementStat(key, val = 1) {
    AppState.stats[key] = (AppState.stats[key] || 0) + val;
    localStorage.setItem('insignia_stats', JSON.stringify(AppState.stats));
    updateStats();
    updateReadiness();
}

function updateReadiness() {
    const s = AppState.stats;
    const rp = Math.min(s.resumes * 25, 100);
    const qp = Math.min(s.questions * 2, 100);
    const tp = Math.min(s.topics * 5, 100);
    const mp = Math.min(s.mocks * 20, 100);
    const overall = Math.round((rp + qp + tp + mp) / 4);

    document.getElementById('resume-progress').style.width = rp + '%';
    document.getElementById('resume-progress-val').textContent = rp + '%';
    document.getElementById('qa-progress').style.width = qp + '%';
    document.getElementById('qa-progress-val').textContent = qp + '%';
    document.getElementById('topic-progress').style.width = tp + '%';
    document.getElementById('topic-progress-val').textContent = tp + '%';
    document.getElementById('mock-progress').style.width = mp + '%';
    document.getElementById('mock-progress-val').textContent = mp + '%';

    document.getElementById('readiness-value').textContent = overall;
    const circle = document.getElementById('readiness-circle');
    const circumference = 327;
    circle.style.strokeDashoffset = circumference - (circumference * overall / 100);
}

// ==================== RESUME BUILDER ====================
function addEducation() {
    const container = document.getElementById('education-entries');
    const idx = container.children.length;
    const block = document.createElement('div');
    block.className = 'entry-block';
    block.innerHTML = `<div class="form-grid"><div class="form-group"><label>Degree</label><input type="text" class="edu-degree" placeholder="M.S. in Data Science"></div><div class="form-group"><label>Institution</label><input type="text" class="edu-institution" placeholder="Stanford University"></div><div class="form-group"><label>Year</label><input type="text" class="edu-year" placeholder="2024 - 2026"></div><div class="form-group"><label>GPA</label><input type="text" class="edu-gpa" placeholder="3.9 / 4.0"></div></div>`;
    container.appendChild(block);
}

function addExperience() {
    const container = document.getElementById('experience-entries');
    const block = document.createElement('div');
    block.className = 'entry-block';
    block.innerHTML = `<div class="form-grid"><div class="form-group"><label>Job Title</label><input type="text" class="exp-title" placeholder="Software Engineer"></div><div class="form-group"><label>Company</label><input type="text" class="exp-company" placeholder="Amazon"></div><div class="form-group"><label>Duration</label><input type="text" class="exp-duration" placeholder="Jan 2024 - Present"></div></div><div class="form-group"><label>Key Responsibilities</label><textarea class="exp-desc" rows="3" placeholder="Describe contributions..."></textarea></div>`;
    container.appendChild(block);
}

function addProject() {
    const container = document.getElementById('project-entries');
    const block = document.createElement('div');
    block.className = 'entry-block';
    block.innerHTML = `<div class="form-grid"><div class="form-group"><label>Project Name</label><input type="text" class="proj-name" placeholder="Project Name"></div><div class="form-group"><label>Technologies</label><input type="text" class="proj-tech" placeholder="React, Python"></div></div><div class="form-group"><label>Description</label><textarea class="proj-desc" rows="2" placeholder="Brief description..."></textarea></div>`;
    container.appendChild(block);
}

async function generateResume(e) {
    e.preventDefault();
    const btn = document.getElementById('generate-resume-btn');
    btn.classList.add('loading');
    btn.textContent = '✨ Generating with AI...';

    const data = {
        name: document.getElementById('resume-name').value,
        email: document.getElementById('resume-email').value,
        phone: document.getElementById('resume-phone').value,
        location: document.getElementById('resume-location').value,
        linkedin: document.getElementById('resume-linkedin').value,
        portfolio: document.getElementById('resume-portfolio').value,
        targetRole: document.getElementById('resume-target-role').value,
        level: document.getElementById('resume-experience-level').value,
        summary: document.getElementById('resume-summary').value,
        skills: document.getElementById('resume-skills').value,
        education: [], experience: [], projects: []
    };

    document.querySelectorAll('#education-entries .entry-block').forEach(block => {
        const deg = block.querySelector('.edu-degree')?.value;
        if (deg) data.education.push({ degree: deg, institution: block.querySelector('.edu-institution')?.value || '', year: block.querySelector('.edu-year')?.value || '', gpa: block.querySelector('.edu-gpa')?.value || '' });
    });
    document.querySelectorAll('#experience-entries .entry-block').forEach(block => {
        const t = block.querySelector('.exp-title')?.value;
        if (t) data.experience.push({ title: t, company: block.querySelector('.exp-company')?.value || '', duration: block.querySelector('.exp-duration')?.value || '', desc: block.querySelector('.exp-desc')?.value || '' });
    });
    document.querySelectorAll('#project-entries .entry-block').forEach(block => {
        const n = block.querySelector('.proj-name')?.value;
        if (n) data.projects.push({ name: n, tech: block.querySelector('.proj-tech')?.value || '', desc: block.querySelector('.proj-desc')?.value || '' });
    });

    // ── Try Python backend first ──────────────────────────────────────────────
    const aiResult = await callBackend('/api/resume/generate', data);

    if (aiResult) {
        // Merge AI-enhanced content back into data
        data.summary = aiResult.enhanced_summary;
        // Replace experience bullet descriptions with AI-enhanced bullets
        if (aiResult.experience_bullets && aiResult.experience_bullets.length) {
            data.experience = data.experience.map((ex, i) => {
                const enhanced = aiResult.experience_bullets[i];
                if (enhanced && enhanced.bullets && enhanced.bullets.length) {
                    ex.desc = enhanced.bullets.join('\n');
                }
                return ex;
            });
        }
        // Store AI achievements for resume layout
        data._aiAchievements = aiResult.key_achievements || [];
        data._aiSuggestions = aiResult.suggestions || [];
        showToast('✨ AI-enhanced resume generated!', 'success');
    } else {
        showToast('Resume generated (offline mode — start Python backend for AI enhancement)', 'info');
    }

    renderResume(data, aiResult);
    btn.classList.remove('loading');
    btn.innerHTML = `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v18M3 12l4-4m0 8l-4-4m18 0l-4 4m0-8l4 4"/></svg> Generate AI Resume`;
    document.getElementById('resume-preview-container').style.display = 'block';
    incrementStat('resumes');
}

function renderResume(d) {
    const enhancedSummary = d.summary || generateSummary(d);
    const skillsArr = d.skills.split(',').map(s => s.trim()).filter(Boolean);

    const expBullets = (desc) => {
        if (!desc) return '';
        const lines = desc.split('\n').filter(l => l.trim());
        return lines.map(l => {
            l = l.replace(/^[-•*]\s*/, '').trim();
            const actionVerbs = ['Developed','Built','Designed','Implemented','Led','Managed','Optimized','Created','Launched','Delivered','Architected','Improved','Reduced','Increased','Spearheaded','Coordinated','Collaborated','Produced','Facilitated','Trained'];
            const hasVerb = actionVerbs.some(v => l.startsWith(v));
            if (!hasVerb && l.length > 3) l = 'Developed ' + l.charAt(0).toLowerCase() + l.slice(1);
            return `<li>${l}</li>`;
        }).join('');
    };

    // Build Key Achievements from experience descriptions
    const achievements = [];
    d.experience.forEach(ex => {
        if (ex.desc) {
            const lines = ex.desc.split('\n').filter(l => l.trim()).slice(0, 1);
            lines.forEach(l => {
                l = l.replace(/^[-•*]\s*/, '').trim();
                if (l) achievements.push({ title: ex.title + ' Achievement', desc: l, company: ex.company });
            });
        }
    });
    if (d.projects.length > 0) {
        d.projects.forEach(p => { if (p.desc) achievements.push({ title: p.name, desc: p.desc, company: p.tech }); });
    }
    if (achievements.length === 0) {
        achievements.push({ title: 'Goal-Oriented Professional', desc: 'Dedicated to delivering high-quality results and continuously improving performance metrics in a dynamic environment.', company: '' });
        achievements.push({ title: 'Continuous Learner', desc: 'Committed to staying current with industry trends and expanding technical expertise to drive innovative solutions.', company: '' });
    }

    // Contacts row
    const contactItems = [];
    if (d.email) contactItems.push(`<span class="cv-contact-item"><span class="cv-contact-icon">✉</span>${d.email}</span>`);
    if (d.linkedin) contactItems.push(`<span class="cv-contact-item"><span class="cv-contact-icon">🔗</span>${d.linkedin}</span>`);
    if (d.portfolio) contactItems.push(`<span class="cv-contact-item"><span class="cv-contact-icon">💼</span>${d.portfolio}</span>`);
    if (d.location) contactItems.push(`<span class="cv-contact-item"><span class="cv-contact-icon">📍</span>${d.location}</span>`);
    if (d.phone) contactItems.push(`<span class="cv-contact-item"><span class="cv-contact-icon">📞</span>${d.phone}</span>`);

    // LEFT: Experience
    let expHtml = '';
    if (d.experience.length) {
        d.experience.forEach(ex => {
            expHtml += `
            <div class="cv-exp-entry">
                <div class="cv-exp-title">${ex.title}</div>
                <div class="cv-exp-company">${ex.company}</div>
                <div class="cv-exp-meta">${ex.duration ? `<span class="cv-exp-icon">📅</span>${ex.duration}` : ''}${d.location ? ` &nbsp;📍 ${d.location}` : ''}</div>
                <ul class="cv-exp-bullets">${expBullets(ex.desc)}</ul>
            </div>`;
        });
    }

    // LEFT: Projects (treated as extra experience)
    let projHtml = '';
    if (d.projects.length) {
        d.projects.forEach(p => {
            projHtml += `
            <div class="cv-exp-entry">
                <div class="cv-exp-title">${p.name}</div>
                <div class="cv-exp-company" style="color:#2ecc71">${p.tech}</div>
                <p class="cv-proj-desc">${p.desc}</p>
            </div>`;
        });
    }

    // RIGHT: Education
    let eduHtml = '';
    if (d.education.length) {
        d.education.forEach(e => {
            eduHtml += `
            <div class="cv-edu-entry">
                <div class="cv-edu-degree">${e.degree}</div>
                <div class="cv-edu-institution">${e.institution}</div>
                <div class="cv-edu-meta">${e.year ? `<span>📅</span> ${e.year}` : ''}${e.gpa ? ` &nbsp; GPA: ${e.gpa}` : ''}</div>
            </div>`;
        });
    }

    // Language bar generator
    const langBar = (level) => {
        const levels = { native: 5, fluent: 4, advanced: 4, intermediate: 3, beginner: 2 };
        const filled = levels[level.toLowerCase()] || 3;
        let bars = '';
        for (let i = 0; i < 5; i++) bars += `<span class="cv-lang-bar ${i < filled ? 'filled' : ''}"></span>`;
        return bars;
    };

    const html = `
    <div class="cv-wrapper">
        <!-- HEADER -->
        <div class="cv-header">
            <div class="cv-header-bg-deco"></div>
            <div class="cv-name">${d.name || 'Your Name'}</div>
            <div class="cv-role">${d.targetRole || 'Professional Title'}</div>
            <div class="cv-contacts">${contactItems.join('')}</div>
        </div>

        <!-- BODY: Two Columns -->
        <div class="cv-body">

            <!-- LEFT COLUMN -->
            <div class="cv-left">

                <!-- SUMMARY -->
                <div class="cv-section">
                    <div class="cv-section-title">SUMMARY</div>
                    <div class="cv-section-line"></div>
                    <p class="cv-summary-text">${enhancedSummary}</p>
                </div>

                <!-- EXPERIENCE -->
                ${d.experience.length ? `
                <div class="cv-section">
                    <div class="cv-section-title">EXPERIENCE</div>
                    <div class="cv-section-line"></div>
                    ${expHtml}
                </div>` : ''}

                <!-- PROJECTS -->
                ${d.projects.length ? `
                <div class="cv-section">
                    <div class="cv-section-title">PROJECTS</div>
                    <div class="cv-section-line"></div>
                    ${projHtml}
                </div>` : ''}

                <!-- LANGUAGES -->
                <div class="cv-section">
                    <div class="cv-section-title">LANGUAGES</div>
                    <div class="cv-section-line"></div>
                    <div class="cv-languages">
                        <div class="cv-lang-item">
                            <div><div class="cv-lang-name">English</div><div class="cv-lang-level">Native</div></div>
                            <div class="cv-lang-bars">${langBar('native')}</div>
                        </div>
                        <div class="cv-lang-item">
                            <div><div class="cv-lang-name">Professional</div><div class="cv-lang-level">Advanced</div></div>
                            <div class="cv-lang-bars">${langBar('advanced')}</div>
                        </div>
                    </div>
                </div>

            </div><!-- end cv-left -->

            <!-- RIGHT COLUMN -->
            <div class="cv-right">

                <!-- KEY ACHIEVEMENTS -->
                <div class="cv-section">
                    <div class="cv-section-title">KEY ACHIEVEMENTS</div>
                    <div class="cv-section-line"></div>
                    ${achievements.slice(0, 4).map(a => `
                    <div class="cv-achievement">
                        <div class="cv-achievement-title">${a.title}</div>
                        <div class="cv-achievement-desc">${a.desc}</div>
                    </div>`).join('')}
                </div>

                <!-- SKILLS -->
                ${skillsArr.length ? `
                <div class="cv-section">
                    <div class="cv-section-title">SKILLS</div>
                    <div class="cv-section-line"></div>
                    <p class="cv-skills-text">${skillsArr.join(', ')}</p>
                </div>` : ''}

                <!-- EDUCATION -->
                ${d.education.length ? `
                <div class="cv-section">
                    <div class="cv-section-title">EDUCATION</div>
                    <div class="cv-section-line"></div>
                    ${eduHtml}
                </div>` : ''}

                <!-- TRAINING / COURSES -->
                <div class="cv-section">
                    <div class="cv-section-title">TRAINING / COURSES</div>
                    <div class="cv-section-line"></div>
                    <div class="cv-training-item">
                        <div class="cv-training-title">${d.targetRole} Fundamentals</div>
                        <div class="cv-training-desc">Industry-standard certification covering core principles and advanced techniques for ${d.targetRole} professionals.</div>
                    </div>
                    <div class="cv-training-item">
                        <div class="cv-training-title">Professional Development</div>
                        <div class="cv-training-desc">Continuous learning through online platforms and industry workshops to stay current with evolving technologies.</div>
                    </div>
                </div>

                <!-- INTERESTS -->
                <div class="cv-section">
                    <div class="cv-section-title">INTERESTS</div>
                    <div class="cv-section-line"></div>
                    <div class="cv-interest-item">
                        <div class="cv-interest-title">Technology Innovation</div>
                        <div class="cv-interest-desc">Passionate about emerging technologies and their potential to solve real-world problems at scale.</div>
                    </div>
                    <div class="cv-interest-item">
                        <div class="cv-interest-title">Continuous Learning</div>
                        <div class="cv-interest-desc">Committed to lifelong growth through courses, open-source contributions, and community engagement.</div>
                    </div>
                </div>

            </div><!-- end cv-right -->
        </div><!-- end cv-body -->

        <!-- FOOTER -->
        <div class="cv-footer">
            <span>Generated by Insignia AI</span>
        </div>
    </div>`;

    document.getElementById('resume-preview').innerHTML = html;
    injectCVStyles();
    generateSuggestions(d);
}

function injectCVStyles() {
    if (document.getElementById('cv-injected-styles')) return;
    const style = document.createElement('style');
    style.id = 'cv-injected-styles';
    style.textContent = `
        #resume-preview { background: #fff; border-radius: 0; padding: 0; font-family: 'Georgia', 'Times New Roman', serif; color: #1a1a2e; }
        .cv-wrapper { max-width: 780px; margin: 0 auto; background: #fff; }
        /* HEADER */
        .cv-header { padding: 28px 32px 20px; background: #fff; position: relative; overflow: hidden; border-bottom: 1px solid #e8e8e8; }
        .cv-header-bg-deco { position: absolute; top: -40px; right: -40px; width: 200px; height: 200px; border-radius: 50%; background: radial-gradient(circle, rgba(108,92,231,0.06) 0%, transparent 70%); pointer-events: none; }
        .cv-name { font-size: 2.1rem; font-weight: 900; letter-spacing: -0.02em; color: #0d0d1a; font-family: 'Arial Black', 'Inter', sans-serif; text-transform: uppercase; line-height: 1.1; }
        .cv-role { font-size: 0.88rem; font-weight: 600; color: #2980b9; margin-top: 4px; letter-spacing: 0.01em; font-family: 'Inter', Arial, sans-serif; }
        .cv-contacts { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }
        .cv-contact-item { font-size: 0.74rem; color: #555; display: flex; align-items: center; gap: 4px; font-family: 'Inter', Arial, sans-serif; }
        .cv-contact-icon { font-size: 0.72rem; }
        /* BODY */
        .cv-body { display: grid; grid-template-columns: 1fr 0.72fr; gap: 0; }
        /* COLUMNS */
        .cv-left { padding: 20px 28px 24px 32px; border-right: 1px solid #ebebeb; }
        .cv-right { padding: 20px 28px 24px 24px; background: #fafafa; }
        /* SECTION */
        .cv-section { margin-bottom: 18px; }
        .cv-section-title { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em; color: #0d0d1a; font-family: 'Inter', Arial, sans-serif; text-transform: uppercase; }
        .cv-section-line { height: 2px; background: #1a1a2e; margin: 4px 0 10px; }
        /* SUMMARY */
        .cv-summary-text { font-size: 0.78rem; color: #444; line-height: 1.6; margin: 0; }
        /* EXPERIENCE */
        .cv-exp-entry { margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px dashed #ddd; }
        .cv-exp-entry:last-child { border-bottom: none; }
        .cv-exp-title { font-size: 0.84rem; font-weight: 700; color: #0d0d1a; font-family: 'Inter', Arial, sans-serif; }
        .cv-exp-company { font-size: 0.78rem; font-weight: 600; color: #2980b9; margin: 2px 0; }
        .cv-exp-meta { font-size: 0.7rem; color: #888; margin-bottom: 6px; display: flex; gap: 4px; align-items: center; font-family: 'Inter', Arial, sans-serif; }
        .cv-exp-icon { font-size: 0.68rem; }
        .cv-exp-bullets { padding-left: 16px; margin: 0; }
        .cv-exp-bullets li { font-size: 0.75rem; color: #555; line-height: 1.55; margin-bottom: 3px; }
        .cv-proj-desc { font-size: 0.75rem; color: #555; margin: 6px 0 0; line-height: 1.55; }
        /* LANGUAGES */
        .cv-languages { display: flex; flex-direction: column; gap: 10px; }
        .cv-lang-item { display: flex; justify-content: space-between; align-items: center; }
        .cv-lang-name { font-size: 0.78rem; font-weight: 700; color: #1a1a2e; font-family: 'Inter', Arial, sans-serif; }
        .cv-lang-level { font-size: 0.68rem; color: #888; }
        .cv-lang-bars { display: flex; gap: 3px; }
        .cv-lang-bar { width: 18px; height: 14px; border-radius: 2px; background: #e0e0e0; }
        .cv-lang-bar.filled { background: #2980b9; }
        /* KEY ACHIEVEMENTS */
        .cv-achievement { margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px dashed #ddd; }
        .cv-achievement:last-child { border-bottom: none; }
        .cv-achievement-title { font-size: 0.76rem; font-weight: 700; color: #0d0d1a; font-family: 'Inter', Arial, sans-serif; margin-bottom: 2px; }
        .cv-achievement-desc { font-size: 0.72rem; color: #555; line-height: 1.5; }
        /* SKILLS */
        .cv-skills-text { font-size: 0.74rem; color: #444; line-height: 1.7; margin: 0; }
        /* EDUCATION */
        .cv-edu-entry { margin-bottom: 10px; }
        .cv-edu-degree { font-size: 0.8rem; font-weight: 700; color: #0d0d1a; font-family: 'Inter', Arial, sans-serif; }
        .cv-edu-institution { font-size: 0.75rem; font-weight: 600; color: #2980b9; }
        .cv-edu-meta { font-size: 0.68rem; color: #888; margin-top: 2px; }
        /* TRAINING */
        .cv-training-item { margin-bottom: 9px; }
        .cv-training-title { font-size: 0.76rem; font-weight: 700; color: #2980b9; font-family: 'Inter', Arial, sans-serif; }
        .cv-training-desc { font-size: 0.71rem; color: #555; line-height: 1.5; margin-top: 2px; }
        /* INTERESTS */
        .cv-interest-item { margin-bottom: 9px; }
        .cv-interest-title { font-size: 0.76rem; font-weight: 700; color: #0d0d1a; font-family: 'Inter', Arial, sans-serif; }
        .cv-interest-desc { font-size: 0.71rem; color: #555; line-height: 1.5; margin-top: 2px; }
        /* FOOTER */
        .cv-footer { text-align: right; padding: 8px 28px; font-size: 0.65rem; color: #bbb; border-top: 1px solid #eee; font-family: 'Inter', Arial, sans-serif; }
    `;
    document.head.appendChild(style);
}

function generateSummary(d) {
    const levels = { fresher: 'motivated and detail-oriented', junior: 'results-driven', mid: 'experienced and innovative', senior: 'seasoned and strategic', lead: 'visionary leader and architect' };
    return `${levels[d.level] || 'Motivated'} ${d.targetRole} with a strong foundation in ${d.skills.split(',').slice(0, 3).join(', ')}. Passionate about delivering high-quality solutions and continuously expanding technical expertise. Seeking to leverage skills and knowledge to drive impactful outcomes in a dynamic environment.`;
}

function generateSuggestions(d) {
    const suggestions = [];
    if (!d.summary) suggestions.push('Add a tailored professional summary highlighting your unique value proposition for the target role.');
    if (d.skills.split(',').length < 5) suggestions.push('Include at least 8-10 relevant skills. Add both technical and soft skills for a balanced profile.');
    if (!d.experience.length) suggestions.push('Add work experience or internships. Even academic projects with real-world impact count.');
    if (d.experience.some(e => !e.desc || e.desc.length < 50)) suggestions.push('Strengthen experience bullet points with quantifiable achievements (e.g., "Improved API response time by 40%").');
    if (!d.linkedin) suggestions.push('Add your LinkedIn profile URL. 87% of recruiters use LinkedIn to vet candidates.');
    if (!d.portfolio) suggestions.push('Include a GitHub or portfolio link to showcase your work and projects.');
    suggestions.push('Use strong action verbs: Architected, Optimized, Spearheaded, Implemented, Delivered.');
    suggestions.push('Tailor keyword density to match the job description for better ATS compatibility.');

    const list = document.getElementById('suggestions-list');
    list.innerHTML = suggestions.map(s => `<li>${s}</li>`).join('');
}

function getResumeCSS() {
    return `
        body { font-family: Georgia, 'Times New Roman', serif; margin: 0; padding: 0; background: #fff; color: #1a1a2e; }
        .cv-wrapper { max-width: 780px; margin: 0 auto; background: #fff; }
        .cv-header { padding: 28px 32px 20px; background: #fff; border-bottom: 1px solid #e8e8e8; }
        .cv-name { font-size: 2.1rem; font-weight: 900; letter-spacing: -0.02em; color: #0d0d1a; font-family: Arial Black, sans-serif; text-transform: uppercase; }
        .cv-role { font-size: 0.88rem; font-weight: 600; color: #2980b9; margin-top: 4px; }
        .cv-contacts { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 10px; }
        .cv-contact-item { font-size: 0.74rem; color: #555; display: flex; align-items: center; gap: 4px; }
        .cv-body { display: grid; grid-template-columns: 1fr 0.72fr; }
        .cv-left { padding: 20px 28px 24px 32px; border-right: 1px solid #ebebeb; }
        .cv-right { padding: 20px 28px 24px 24px; background: #fafafa; }
        .cv-section { margin-bottom: 18px; }
        .cv-section-title { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em; color: #0d0d1a; text-transform: uppercase; }
        .cv-section-line { height: 2px; background: #1a1a2e; margin: 4px 0 10px; }
        .cv-summary-text { font-size: 0.78rem; color: #444; line-height: 1.6; margin: 0; }
        .cv-exp-entry { margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px dashed #ddd; }
        .cv-exp-title { font-size: 0.84rem; font-weight: 700; color: #0d0d1a; }
        .cv-exp-company { font-size: 0.78rem; font-weight: 600; color: #2980b9; margin: 2px 0; }
        .cv-exp-meta { font-size: 0.7rem; color: #888; margin-bottom: 6px; }
        .cv-exp-bullets { padding-left: 16px; margin: 0; }
        .cv-exp-bullets li { font-size: 0.75rem; color: #555; line-height: 1.55; margin-bottom: 3px; }
        .cv-proj-desc { font-size: 0.75rem; color: #555; margin: 6px 0 0; line-height: 1.55; }
        .cv-languages { display: flex; flex-direction: column; gap: 10px; }
        .cv-lang-item { display: flex; justify-content: space-between; align-items: center; }
        .cv-lang-name { font-size: 0.78rem; font-weight: 700; }
        .cv-lang-level { font-size: 0.68rem; color: #888; }
        .cv-lang-bars { display: flex; gap: 3px; }
        .cv-lang-bar { width: 18px; height: 14px; border-radius: 2px; background: #e0e0e0; }
        .cv-lang-bar.filled { background: #2980b9; }
        .cv-achievement { margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px dashed #ddd; }
        .cv-achievement-title { font-size: 0.76rem; font-weight: 700; margin-bottom: 2px; }
        .cv-achievement-desc { font-size: 0.72rem; color: #555; line-height: 1.5; }
        .cv-skills-text { font-size: 0.74rem; color: #444; line-height: 1.7; margin: 0; }
        .cv-edu-degree { font-size: 0.8rem; font-weight: 700; }
        .cv-edu-institution { font-size: 0.75rem; font-weight: 600; color: #2980b9; }
        .cv-edu-meta { font-size: 0.68rem; color: #888; margin-top: 2px; }
        .cv-training-title { font-size: 0.76rem; font-weight: 700; color: #2980b9; }
        .cv-training-desc { font-size: 0.71rem; color: #555; line-height: 1.5; margin-top: 2px; }
        .cv-interest-title { font-size: 0.76rem; font-weight: 700; }
        .cv-interest-desc { font-size: 0.71rem; color: #555; line-height: 1.5; margin-top: 2px; }
        .cv-footer { text-align: right; padding: 8px 28px; font-size: 0.65rem; color: #bbb; border-top: 1px solid #eee; }
    `;
}

function downloadResume() {
    const content = document.getElementById('resume-preview').innerHTML;
    const blob = new Blob([`<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Resume</title><style>${getResumeCSS()}</style></head><body>${content}</body></html>`], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'resume.html'; a.click();
    URL.revokeObjectURL(url);
    showToast('Resume downloaded!', 'success');
}

// ==================== Q&A GENERATOR ====================
const QA_DATABASE = {
    technical: {
        'Software Engineer': [
            { q: "Explain the difference between a stack and a queue. When would you use each?", a: "A stack follows LIFO (Last In, First Out) — the last element added is the first removed. A queue follows FIFO (First In, First Out). Use stacks for undo operations, expression parsing, and DFS. Use queues for task scheduling, BFS, and buffering.", criteria: "Understanding of data structure fundamentals, practical application examples" },
            { q: "What is the time complexity of common sorting algorithms, and when would you choose one over another?", a: "Quick Sort: O(n log n) average, O(n²) worst — great general-purpose. Merge Sort: O(n log n) guaranteed — ideal for linked lists and stable sorting. Heap Sort: O(n log n) — good for memory-constrained situations. For small arrays (<10 elements), Insertion Sort can outperform due to low overhead.", criteria: "Algorithmic knowledge, ability to compare trade-offs" },
            { q: "Explain REST API design principles and best practices.", a: "REST APIs should be stateless, use proper HTTP methods (GET, POST, PUT, DELETE), return appropriate status codes, version endpoints, use consistent naming conventions, implement pagination for large datasets, and include proper error handling with meaningful messages.", criteria: "API design understanding, awareness of standards and best practices" },
            { q: "What are SOLID principles in object-oriented design?", a: "S: Single Responsibility — one class, one reason to change. O: Open/Closed — open for extension, closed for modification. L: Liskov Substitution — subtypes must be substitutable. I: Interface Segregation — prefer small, specific interfaces. D: Dependency Inversion — depend on abstractions, not concretions.", criteria: "OOP design understanding, ability to apply in real scenarios" },
            { q: "Describe how a hash table works internally. How do you handle collisions?", a: "A hash table uses a hash function to map keys to array indices. Collisions are handled via chaining (linked lists at each index) or open addressing (probing for next available slot). Load factor affects performance — typically rehash when it exceeds 0.75.", criteria: "Deep understanding of hash tables, collision resolution strategies" }
        ],
        'Frontend Developer': [
            { q: "Explain the Virtual DOM and how React uses it for efficient rendering.", a: "The Virtual DOM is a lightweight JS representation of the actual DOM. React creates a virtual copy, applies changes there first, then diffs (reconciliation) to find minimal DOM updates needed. This batch processing is more efficient than direct DOM manipulation.", criteria: "Understanding of React internals, performance optimization knowledge" },
            { q: "What is the difference between CSS Grid and Flexbox? When would you use each?", a: "Flexbox is one-dimensional (row OR column) — ideal for components, navigation bars, centering. Grid is two-dimensional (rows AND columns) — ideal for page layouts, complex grid systems. They complement each other and can be nested.", criteria: "CSS layout mastery, practical application judgment" },
            { q: "Explain event delegation in JavaScript and why it's useful.", a: "Event delegation uses event bubbling to handle events at a parent level rather than attaching listeners to each child. Benefits: fewer event listeners (better memory), automatically handles dynamically added elements, cleaner code.", criteria: "DOM event model understanding, performance awareness" },
            { q: "What are Web Vitals and how do you optimize for them?", a: "Core Web Vitals: LCP (Largest Contentful Paint <2.5s), FID (First Input Delay <100ms), CLS (Cumulative Layout Shift <0.1). Optimize via: lazy loading, code splitting, image optimization, font preloading, proper element sizing, and efficient JavaScript execution.", criteria: "Performance metrics knowledge, optimization strategies" },
            { q: "Explain closures in JavaScript with a practical example.", a: "A closure is a function that retains access to variables from its outer scope even after the outer function has returned. Example: function createCounter() { let count = 0; return () => ++count; }. The inner function 'closes over' the count variable.", criteria: "Fundamental JS concept understanding, practical demonstration" }
        ],
        'Data Scientist': [
            { q: "Explain the bias-variance tradeoff in machine learning.", a: "Bias is error from overly simple models (underfitting). Variance is error from overly complex models (overfitting). The tradeoff: reducing bias increases variance and vice versa. The goal is finding the sweet spot that minimizes total error. Techniques: cross-validation, regularization, ensemble methods.", criteria: "Core ML theory understanding, practical mitigation strategies" },
            { q: "When would you use Random Forest vs Gradient Boosting?", a: "Random Forest: parallel trees, robust to outliers, less prone to overfitting, faster training. Gradient Boosting: sequential trees, often higher accuracy, requires careful tuning, handles complex patterns better. Use RF for quick baselines; GB (XGBoost/LightGBM) when accuracy is paramount.", criteria: "Algorithm comparison skills, practical decision-making" },
            { q: "How do you handle missing data in a dataset?", a: "Strategies: 1) Remove rows/columns (if <5% missing). 2) Mean/median/mode imputation. 3) KNN imputation. 4) MICE (Multiple Imputation). 5) Domain-specific logic. 6) Create a 'missing' indicator feature. Choice depends on data mechanism (MCAR, MAR, MNAR) and downstream model.", criteria: "Data preprocessing knowledge, statistical understanding" },
            { q: "Explain precision, recall, and F1-score. When is each most important?", a: "Precision = TP/(TP+FP) — when false positives are costly (spam detection). Recall = TP/(TP+FN) — when false negatives are costly (cancer detection). F1 = harmonic mean of both — balanced metric. Use PR-AUC for imbalanced datasets.", criteria: "Metrics understanding, domain-appropriate metric selection" },
            { q: "What is feature engineering and why is it important?", a: "Feature engineering is creating new input features from existing data to improve model performance. Techniques: polynomial features, binning, encoding categoricals, text vectorization, date decomposition, domain-specific aggregations. Often more impactful than algorithm selection.", criteria: "Practical data science skills, creativity in feature creation" }
        ]
    },
    behavioral: [
        { q: "Tell me about a time you faced a challenging technical problem. How did you solve it?", a: "Use the STAR method: Describe the Situation and Task clearly. Detail the specific Actions you took — breaking down the problem, researching solutions, collaborating with team members. Share the Result with measurable outcomes. Emphasize your problem-solving methodology.", criteria: "STAR method, problem-solving approach, persistence" },
        { q: "Describe a situation where you had to work with a difficult team member.", a: "Focus on empathy, communication, and professionalism. Describe how you sought to understand their perspective, found common ground, established clear communication channels, and worked toward shared goals. Emphasize the positive outcome and what you learned about collaboration.", criteria: "Emotional intelligence, conflict resolution, teamwork" },
        { q: "How do you handle tight deadlines and competing priorities?", a: "Discuss prioritization frameworks (Eisenhower matrix), time management techniques, proactive communication with stakeholders about tradeoffs, and examples of successfully delivering under pressure. Show that you can maintain quality while managing time constraints.", criteria: "Time management, prioritization skills, stress handling" },
        { q: "Tell me about a time you made a mistake at work. How did you handle it?", a: "Show accountability — don't deflect. Describe the mistake, how you identified it, immediate steps to mitigate impact, what you communicated to the team, and concrete measures you implemented to prevent recurrence. Focus on growth and learning.", criteria: "Accountability, learning mindset, transparency" },
        { q: "Where do you see yourself in 5 years?", a: "Align your goals with the company's growth trajectory. Show ambition while being realistic. Discuss skill development, leadership aspirations, and desire to make meaningful contributions. Avoid overly specific titles; focus on impact and growth areas.", criteria: "Self-awareness, career planning, alignment with company vision" },
        { q: "Why are you interested in this role/company?", a: "Research the company thoroughly. Connect your skills and passions to the company's mission, products, and culture. Be specific about what excites you. Mention recent company achievements or initiatives that resonate with your values.", criteria: "Research depth, genuine enthusiasm, cultural fit" }
    ],
    situational: [
        { q: "If you discovered a critical bug in production on a Friday evening, what would you do?", a: "Immediately assess severity and impact scope. Communicate to relevant stakeholders (manager, on-call team). If possible, implement a quick fix or rollback. Document the incident. Schedule a post-mortem for root cause analysis. Prioritize user safety and data integrity.", criteria: "Crisis management, communication, technical judgment" },
        { q: "How would you approach learning a completely new technology stack for a project?", a: "Start with official documentation and tutorials. Build a small prototype to get hands-on experience. Join community forums and study best practices. Identify parallels with known technologies. Set learning milestones and seek mentorship. Balance learning with delivery timelines.", criteria: "Learning agility, resourcefulness, structured approach" },
        { q: "If your team disagrees on the technical approach for a feature, how would you resolve it?", a: "Facilitate a structured discussion where each approach is evaluated against objective criteria (performance, maintainability, timeline). Create proof-of-concepts if needed. Consider team expertise and project constraints. Seek consensus but be willing to make a decision and commit.", criteria: "Leadership, objectivity, decision-making" }
    ]
};

function getQuestionsForRole(role, type, count, level) {
    let questions = [];
    const roleKey = Object.keys(QA_DATABASE.technical).find(k => k.toLowerCase().includes(role.toLowerCase())) || Object.keys(QA_DATABASE.technical)[0];
    const techQs = (QA_DATABASE.technical[roleKey] || QA_DATABASE.technical['Software Engineer']).map(q => ({ ...q, type: 'technical' }));
    const behQs = QA_DATABASE.behavioral.map(q => ({ ...q, type: 'behavioral' }));
    const sitQs = QA_DATABASE.situational.map(q => ({ ...q, type: 'situational' }));

    if (type === 'technical') questions = techQs;
    else if (type === 'behavioral') questions = behQs;
    else if (type === 'situational') questions = sitQs;
    else questions = [...techQs, ...behQs, ...sitQs];

    // Shuffle
    for (let i = questions.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [questions[i], questions[j]] = [questions[j], questions[i]]; }
    return questions.slice(0, parseInt(count));
}

async function generateQuestions() {
    const role = document.getElementById('qa-role').value;
    if (!role) { showToast('Please enter a target role', 'error'); return; }
    const level = document.getElementById('qa-level').value;
    const type = document.getElementById('qa-type').value;
    const count = parseInt(document.getElementById('qa-count').value);

    const btn = document.getElementById('generate-qa-btn');
    btn.classList.add('loading');
    btn.textContent = '🤖 Generating with AI...';

    // ── Try Python backend first ──────────────────────────────────────────────
    const aiQuestions = await callBackend('/api/qa/generate', { role, level, type, count });

    if (aiQuestions && aiQuestions.length) {
        // Backend returns: { question, model_answer, type, criteria }
        AppState.qaQuestions = aiQuestions;
        renderQuestions(aiQuestions, true);
        incrementStat('questions', aiQuestions.length);
        showToast(`✨ Generated ${aiQuestions.length} AI questions for ${role}!`, 'success');
    } else {
        // Fallback to local database
        const questions = getQuestionsForRole(role, type, count, level);
        AppState.qaQuestions = questions;
        renderQuestions(questions, false);
        incrementStat('questions', questions.length);
        showToast(`Generated ${questions.length} questions (offline mode)`, 'info');
    }

    btn.classList.remove('loading');
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3v18M3 12l4-4m0 8l-4-4m18 0l-4 4m0-8l4 4"/></svg> Generate Questions`;
}

function renderQuestions(questions, fromBackend = false) {
    const container = document.getElementById('qa-container');
    container.innerHTML = questions.map((q, i) => {
        // Support both backend schema (question/model_answer) and local schema (q/a)
        const questionText = q.question || q.q || '';
        const answerText   = q.model_answer || q.a || '';
        const qType        = q.type || 'technical';
        const qCriteria    = q.criteria || 'Clarity, depth, and relevance.';
        return `
        <div class="qa-card" id="qa-card-${i}">
            <div class="qa-card-header" onclick="toggleQA(${i})">
                <div class="qa-number">${i + 1}</div>
                <div class="qa-question">${questionText}</div>
                <span class="qa-type-badge ${qType}">${qType}</span>
                ${fromBackend ? '<span class="qa-ai-badge">✨ AI</span>' : ''}
                <div class="qa-toggle"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></div>
            </div>
            <div class="qa-body">
                <div class="qa-section-label">Model Answer</div>
                <div class="qa-answer">${answerText}</div>
                <div class="qa-section-label">Evaluation Criteria</div>
                <div class="qa-criteria">${qCriteria}</div>
                <div class="qa-section-label">Practice Your Answer</div>
                <div class="qa-practice-area">
                    <textarea id="qa-practice-${i}" placeholder="Type your answer here to get AI feedback..."></textarea>
                    <div class="qa-practice-actions">
                        <button class="btn btn-sm btn-primary" onclick="evaluateAnswer(${i}, '${qType}')">Get AI Feedback</button>
                    </div>
                    <div id="qa-feedback-${i}"></div>
                </div>
            </div>
        </div>`;
    }).join('');
}

function toggleQA(idx) {
    document.getElementById('qa-card-' + idx).classList.toggle('open');
}

async function evaluateAnswer(idx, type) {
    const answer = document.getElementById('qa-practice-' + idx).value;
    if (!answer || answer.length < 20) { showToast('Please write a more detailed answer', 'error'); return; }

    const feedbackEl = document.getElementById('qa-feedback-' + idx);
    feedbackEl.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div><span style="font-size:0.8rem;color:var(--text-muted);margin-left:8px">AI is evaluating your answer...</span>';

    // Get the question text for context
    const qData = AppState.qaQuestions[idx];
    const questionText = qData ? (qData.question || qData.q || '') : '';
    const role = document.getElementById('qa-role')?.value || 'Professional';

    // ── Try Python backend ────────────────────────────────────────────────────
    const aiResult = await callBackend('/api/mock/feedback', {
        question: questionText,
        question_type: type,
        user_answer: answer,
        role
    });

    if (aiResult) {
        const gradeColor = aiResult.score >= 85 ? '#00B894' : aiResult.score >= 70 ? '#6C5CE7' : aiResult.score >= 50 ? '#FDCB6E' : '#FF6B6B';
        feedbackEl.innerHTML = `
        <div class="qa-feedback ai-feedback">
            <div class="ai-score-row">
                <div class="ai-score-badge" style="background:${gradeColor}">${aiResult.score}/100 — ${aiResult.grade}</div>
                <span class="ai-powered-tag">✨ Gemini AI</span>
            </div>
            ${aiResult.strengths.length ? `
            <div class="ai-feedback-section">
                <div class="ai-fb-label">✅ Strengths</div>
                <ul>${aiResult.strengths.map(s => `<li>${s}</li>`).join('')}</ul>
            </div>` : ''}
            ${aiResult.improvements.length ? `
            <div class="ai-feedback-section">
                <div class="ai-fb-label">🔧 Areas to Improve</div>
                <ul>${aiResult.improvements.map(s => `<li>${s}</li>`).join('')}</ul>
            </div>` : ''}
            <div class="ai-feedback-section">
                <div class="ai-fb-label">💡 Model Answer Snippet</div>
                <div class="ai-model-snippet">${aiResult.model_answer_snippet}</div>
            </div>
            <div class="ai-overall">${aiResult.overall_feedback}</div>
        </div>`;
    } else {
        // Fallback local scoring
        const wordCount = answer.split(/\s+/).length;
        const score = Math.min(95, Math.max(40, 50 + wordCount + Math.floor(Math.random() * 20)));
        let feedback = '';
        if (score >= 80) feedback = `<strong>Score: ${score}/100 — Excellent!</strong><br>Your answer demonstrates strong understanding. `;
        else if (score >= 60) feedback = `<strong>Score: ${score}/100 — Good</strong><br>Solid foundation. `;
        else feedback = `<strong>Score: ${score}/100 — Needs Improvement</strong><br>`;
        if (wordCount < 30) feedback += 'Consider expanding your answer with more specific examples. ';
        if (type === 'behavioral' && !answer.toLowerCase().includes('result')) feedback += 'Include specific Results using the STAR method. ';
        if (type === 'technical' && wordCount < 50) feedback += 'Add more technical depth with specific tools or patterns. ';
        feedback += 'Start the Python backend for detailed AI feedback!';
        feedbackEl.innerHTML = `<div class="qa-feedback">${feedback}</div>`;
    }
}

// ==================== STUDY HUB ====================
const STUDY_DATA = {
    'frontend': {
        title: 'Frontend Developer',
        modules: [
            { title: 'HTML & CSS Fundamentals', topics: [
                { name: 'Semantic HTML5', desc: 'Use semantic elements (article, section, nav, header, footer) for better accessibility and SEO.', points: ['Block vs inline elements and the box model', 'Forms, inputs, and validation', 'Accessibility (ARIA roles, alt text, keyboard nav)'] },
                { name: 'CSS Layout Systems', desc: 'Master modern layout techniques for responsive, maintainable designs.', points: ['Flexbox: alignment, ordering, wrapping', 'CSS Grid: template areas, auto-fit, minmax', 'Responsive design with media queries', 'CSS custom properties (variables)'] },
                { name: 'CSS Animations & Transitions', desc: 'Create smooth, performant animations.', points: ['Transitions vs keyframe animations', 'Transform and opacity for GPU acceleration', 'Reduced motion preferences'] }
            ]},
            { title: 'JavaScript Core', topics: [
                { name: 'ES6+ Features', desc: 'Modern JavaScript syntax and capabilities.', points: ['Arrow functions, destructuring, spread/rest', 'Promises, async/await, error handling', 'Modules (import/export)', 'Optional chaining, nullish coalescing'] },
                { name: 'DOM Manipulation', desc: 'Interact with and modify the document.', points: ['Event handling and delegation', 'DOM traversal and manipulation', 'IntersectionObserver, MutationObserver', 'Web APIs (Fetch, Storage, History)'] },
                { name: 'Closures & Scope', desc: 'Understand execution context.', points: ['Lexical scope and scope chain', 'Closures and practical applications', 'this keyword and binding', 'Event loop and microtasks'] }
            ]},
            { title: 'React Ecosystem', topics: [
                { name: 'React Fundamentals', desc: 'Core React concepts.', points: ['JSX, components, props, state', 'Hooks: useState, useEffect, useRef, useMemo', 'Context API and state management', 'React Router and navigation'] },
                { name: 'State Management', desc: 'Managing complex application state.', points: ['Redux Toolkit and RTK Query', 'Zustand, Jotai, Recoil alternatives', 'Server state: React Query / SWR', 'When to use which approach'] },
                { name: 'Performance Optimization', desc: 'Build fast React applications.', points: ['React.memo, useMemo, useCallback', 'Code splitting and lazy loading', 'Virtual scrolling for large lists', 'Profiling and DevTools usage'] }
            ]},
            { title: 'Build Tools & Testing', topics: [
                { name: 'Module Bundlers', desc: 'Understand build tooling.', points: ['Vite, Webpack, esbuild', 'Tree shaking and code splitting', 'Environment variables and configs'] },
                { name: 'Testing', desc: 'Write reliable tests.', points: ['Jest / Vitest for unit testing', 'React Testing Library', 'E2E with Playwright/Cypress', 'Testing patterns and best practices'] }
            ]}
        ]
    },
    'backend': {
        title: 'Backend Developer',
        modules: [
            { title: 'Server & APIs', topics: [
                { name: 'RESTful API Design', desc: 'Design robust, scalable APIs.', points: ['HTTP methods, status codes, headers', 'Resource naming conventions', 'Pagination, filtering, sorting', 'API versioning strategies'] },
                { name: 'Authentication & Authorization', desc: 'Secure your APIs.', points: ['JWT tokens and refresh flow', 'OAuth 2.0 and OpenID Connect', 'Role-based access control (RBAC)', 'API keys and rate limiting'] }
            ]},
            { title: 'Databases', topics: [
                { name: 'SQL Databases', desc: 'Relational database concepts.', points: ['JOINS, subqueries, CTEs', 'Indexing strategies and query optimization', 'Transactions and ACID properties', 'Database normalization (1NF-3NF)'] },
                { name: 'NoSQL & Caching', desc: 'Non-relational data stores.', points: ['MongoDB document modeling', 'Redis caching strategies', 'When SQL vs NoSQL', 'CAP theorem and consistency models'] }
            ]},
            { title: 'System Design', topics: [
                { name: 'Scalability', desc: 'Design for scale.', points: ['Horizontal vs vertical scaling', 'Load balancing strategies', 'Microservices vs monolith', 'Message queues (Kafka, RabbitMQ)'] },
                { name: 'Design Patterns', desc: 'Common backend patterns.', points: ['Repository, Factory, Observer', 'CQRS and Event Sourcing', 'Circuit Breaker, Retry', 'Dependency Injection'] }
            ]}
        ]
    },
    'data-science': {
        title: 'Data Scientist',
        modules: [
            { title: 'Statistics & Probability', topics: [
                { name: 'Descriptive Statistics', desc: 'Summarize and understand data.', points: ['Central tendency and dispersion', 'Distributions (Normal, Poisson, Binomial)', 'Correlation vs causation', 'Hypothesis testing (t-test, chi-square, ANOVA)'] },
                { name: 'Probability Theory', desc: 'Foundation for ML.', points: ['Bayes theorem and applications', 'Conditional probability', 'Random variables and expectations', 'Central Limit Theorem'] }
            ]},
            { title: 'Machine Learning', topics: [
                { name: 'Supervised Learning', desc: 'Learn from labeled data.', points: ['Linear/Logistic Regression', 'Decision Trees, Random Forest, XGBoost', 'SVM, KNN, Naive Bayes', 'Evaluation metrics and cross-validation'] },
                { name: 'Unsupervised Learning', desc: 'Discover hidden patterns.', points: ['K-Means, DBSCAN clustering', 'PCA and dimensionality reduction', 'Association rules', 'Anomaly detection'] },
                { name: 'Deep Learning', desc: 'Neural network architectures.', points: ['Neural network fundamentals', 'CNNs for computer vision', 'RNNs/LSTMs for sequences', 'Transformers and attention mechanism'] }
            ]},
            { title: 'Data Engineering', topics: [
                { name: 'Data Wrangling', desc: 'Clean and prepare data.', points: ['Pandas and NumPy mastery', 'Handling missing data strategies', 'Feature engineering techniques', 'Data pipeline creation'] }
            ]}
        ]
    }
};

// Generate generic data for roles not in STUDY_DATA
function getGenericStudyData(domain) {
    return { title: domain.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase()), modules: [
        { title: 'Core Concepts', topics: [{ name: 'Fundamentals', desc: 'Master the foundational concepts.', points: ['Core principles and terminology', 'Industry standards and best practices', 'Common tools and frameworks', 'Practical applications'] }] },
        { title: 'Advanced Topics', topics: [{ name: 'Advanced Techniques', desc: 'Deep dive into specialized areas.', points: ['Latest industry trends', 'Performance optimization', 'Architecture patterns', 'Case studies and real-world scenarios'] }] },
        { title: 'Interview Preparation', topics: [{ name: 'Common Interview Topics', desc: 'Frequently asked areas.', points: ['System design questions', 'Problem-solving scenarios', 'Behavioral question preparation', 'Portfolio and project discussion'] }] }
    ]};
}

async function generateStudyPlan() {
    const domain = document.getElementById('study-domain').value;
    if (!domain) { showToast('Please select a domain', 'error'); return; }
    const level = document.getElementById('study-level').value;

    const btn = document.querySelector('[onclick="generateStudyPlan()"]');
    if (btn) { btn.classList.add('loading'); btn.textContent = '🤖 AI Generating...'; }

    // ── Try Python backend ────────────────────────────────────────────────────
    const aiModules = await callBackend('/api/study/generate', { domain, level });

    if (aiModules && aiModules.length) {
        renderStudyPlanFromBackend(aiModules, domain);
        showToast(`✨ AI study plan generated for ${domain}!`, 'success');
    } else {
        // Fallback to local
        const data = STUDY_DATA[domain] || getGenericStudyData(domain);
        renderStudyPlan(data);
        showToast(`Study plan generated (offline mode)`, 'info');
    }

    if (btn) {
        btn.classList.remove('loading');
        btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg> Generate Study Plan`;
    }
}

function renderStudyPlanFromBackend(modules, domain) {
    // Backend schema: { title, topics: [{ name, description, key_points, resources }] }
    const container = document.getElementById('study-content');
    container.innerHTML = modules.map((mod, mi) => `
        <div class="study-module ${mi === 0 ? 'open' : ''}" id="study-mod-${mi}">
            <div class="study-module-header" onclick="toggleStudyModule(${mi})">
                <div class="study-module-number">${mi + 1}</div>
                <div class="study-module-info">
                    <div class="study-module-title">${mod.title}</div>
                    <div class="study-module-meta">${mod.topics.length} topics &nbsp;✨ AI</div>
                </div>
                <div class="study-module-toggle"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></div>
            </div>
            <div class="study-module-body">
                ${mod.topics.map((topic, ti) => `
                    <div class="study-topic">
                        <h4>${topic.name}</h4>
                        <p>${topic.description}</p>
                        <ul class="key-points">${(topic.key_points || []).map(p => `<li>${p}</li>`).join('')}</ul>
                        ${(topic.resources || []).length ? `
                        <div class="study-resources">
                            <div class="resource-label">📚 Resources</div>
                            <ul class="resource-list">${topic.resources.map(r => `<li>${r}</li>`).join('')}</ul>
                        </div>` : ''}
                        <label class="study-check ${isTopicCompleted(mi, ti) ? 'completed' : ''}">
                            <input type="checkbox" ${isTopicCompleted(mi, ti) ? 'checked' : ''} onchange="toggleTopicCompletion(${mi}, ${ti}, this)">
                            Mark as completed
                        </label>
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');
}

function renderStudyPlan(data) {
    const container = document.getElementById('study-content');
    container.innerHTML = data.modules.map((mod, mi) => `
        <div class="study-module ${mi === 0 ? 'open' : ''}" id="study-mod-${mi}">
            <div class="study-module-header" onclick="toggleStudyModule(${mi})">
                <div class="study-module-number">${mi + 1}</div>
                <div class="study-module-info">
                    <div class="study-module-title">${mod.title}</div>
                    <div class="study-module-meta">${mod.topics.length} topics</div>
                </div>
                <div class="study-module-toggle"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg></div>
            </div>
            <div class="study-module-body">
                ${mod.topics.map((topic, ti) => `
                    <div class="study-topic">
                        <h4>${topic.name}</h4>
                        <p>${topic.desc}</p>
                        <ul class="key-points">${topic.points.map(p => `<li>${p}</li>`).join('')}</ul>
                        <label class="study-check ${isTopicCompleted(mi, ti) ? 'completed' : ''}">
                            <input type="checkbox" ${isTopicCompleted(mi, ti) ? 'checked' : ''} onchange="toggleTopicCompletion(${mi}, ${ti}, this)">
                            Mark as completed
                        </label>
                    </div>
                `).join('')}
            </div>
        </div>
    `).join('');
}

function toggleStudyModule(idx) { document.getElementById('study-mod-' + idx).classList.toggle('open'); }

function isTopicCompleted(mi, ti) { return AppState.studyProgress[`${mi}-${ti}`] === true; }

function toggleTopicCompletion(mi, ti, el) {
    const key = `${mi}-${ti}`;
    AppState.studyProgress[key] = el.checked;
    localStorage.setItem('insignia_study_progress', JSON.stringify(AppState.studyProgress));
    el.parentElement.classList.toggle('completed', el.checked);
    if (el.checked) incrementStat('topics');
}

// ==================== VOICE RECOGNITION & SYNTHESIS ====================
let recognition = null;
let isRecording = false;

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    
    recognition.onresult = (event) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            }
        }
        if (finalTranscript) {
            const input = document.getElementById('mock-answer-input');
            const currentVal = input.value;
            input.value = currentVal + (currentVal.endsWith(' ') || currentVal === '' ? '' : ' ') + finalTranscript;
        }
    };
    
    recognition.onerror = (event) => {
        console.warn('Speech recognition error', event.error);
        if(event.error !== 'no-speech') {
            stopMockMic();
        }
    };
    
    recognition.onend = () => {
        stopMockMic();
    };
}

function toggleMockMic(e) {
    if(e) e.preventDefault();
    if (!recognition) {
        showToast('Voice input is not supported in this browser', 'error');
        return;
    }
    
    if (isRecording) {
        stopMockMic();
    } else {
        startMockMic();
    }
}

function startMockMic() {
    if (!recognition) return;
    try {
        recognition.start();
        isRecording = true;
        const btn = document.getElementById('mock-mic-btn');
        if(btn) {
            btn.classList.add('recording-active');
            btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="red" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect></svg> Stop Voice Input`;
            btn.style.borderColor = 'red';
            btn.style.color = 'red';
        }
    } catch(err) {
        console.log('Recognition already started', err);
    }
}

function stopMockMic() {
    if (!recognition) return;
    try { recognition.stop(); } catch(e){}
    isRecording = false;
    const btn = document.getElementById('mock-mic-btn');
    if(btn) {
        btn.classList.remove('recording-active');
        btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="22"></line></svg> Start Voice Input`;
        btn.style.borderColor = '';
        btn.style.color = '';
    }
}

function readMockQuestion() {
    if (!('speechSynthesis' in window)) return;
    
    window.speechSynthesis.cancel();
    const text = document.getElementById('mock-question-text').textContent;
    if (!text || text === 'Loading question...') return;
    
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    const voice = voices.find(v => v.lang.startsWith('en-') && (v.name.includes('Google') || v.name.includes('Natural'))) || voices.find(v => v.lang.startsWith('en-'));
    if (voice) utterance.voice = voice;
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
}

function stopMockSpeaking() {
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }
}

// ==================== MOCK INTERVIEW ====================
function startMockInterview() {
    const role = document.getElementById('mock-role').value;
    if (!role) { showToast('Please enter a target role', 'error'); return; }
    const type = document.getElementById('mock-type').value;
    const count = parseInt(document.getElementById('mock-questions-count').value);
    const questions = getQuestionsForRole(role, type === 'mixed' ? 'all' : type, count);

    AppState.mockSession = { role, questions, currentIndex: 0, answers: [], startTime: Date.now() };
    AppState.mockSeconds = 0;

    document.getElementById('mock-setup').style.display = 'none';
    document.getElementById('mock-session').style.display = 'block';
    document.getElementById('mock-results').style.display = 'none';

    startMockTimer();
    showMockQuestion();
}

function startMockTimer() {
    clearInterval(AppState.mockTimer);
    AppState.mockTimer = setInterval(() => {
        AppState.mockSeconds++;
        const mins = Math.floor(AppState.mockSeconds / 60).toString().padStart(2, '0');
        const secs = (AppState.mockSeconds % 60).toString().padStart(2, '0');
        document.getElementById('mock-timer').textContent = `${mins}:${secs}`;
    }, 1000);
}

function showMockQuestion() {
    const s = AppState.mockSession;
    const q = s.questions[s.currentIndex];
    const total = s.questions.length;

    document.getElementById('mock-question-counter').textContent = `Question ${s.currentIndex + 1} of ${total}`;
    document.getElementById('mock-progress-fill').style.width = `${((s.currentIndex) / total) * 100}%`;
    document.getElementById('mock-q-type').textContent = q.type;
    document.getElementById('mock-question-text').textContent = q.q;
    document.getElementById('mock-answer-input').value = '';
    document.getElementById('mock-answer-input').focus();
    
    // Voice extensions
    stopMockMic();
    setTimeout(readMockQuestion, 300);
}

function submitMockAnswer() {
    const answer = document.getElementById('mock-answer-input').value;
    if (!answer.trim()) { showToast('Please type an answer before submitting', 'error'); return; }
    stopMockMic();
    stopMockSpeaking();
    AppState.mockSession.answers.push({ answer, skipped: false });
    advanceMock();
}

function skipMockQuestion() {
    stopMockMic();
    stopMockSpeaking();
    AppState.mockSession.answers.push({ answer: '(Skipped)', skipped: true });
    advanceMock();
}

function advanceMock() {
    const s = AppState.mockSession;
    s.currentIndex++;
    if (s.currentIndex >= s.questions.length) {
        finishMockInterview();
    } else {
        showMockQuestion();
    }
}

function finishMockInterview() {
    clearInterval(AppState.mockTimer);
    document.getElementById('mock-session').style.display = 'none';
    document.getElementById('mock-results').style.display = 'block';
    
    stopMockMic();
    stopMockSpeaking();

    const s = AppState.mockSession;
    const results = s.questions.map((q, i) => {
        const ans = s.answers[i];
        if (ans.skipped) return { ...q, answer: ans.answer, score: 0, feedback: 'Question was skipped. Attempting all questions shows confidence and willingness to try.' };
        const wc = ans.answer.split(/\s+/).length;
        const score = Math.min(95, Math.max(30, 45 + wc + Math.floor(Math.random() * 25)));
        let feedback = '';
        if (score >= 80) feedback = 'Excellent response! Demonstrates strong knowledge and clear communication. ';
        else if (score >= 60) feedback = 'Good effort. ';
        else feedback = 'Needs more depth. ';
        if (wc < 25) feedback += 'Try to elaborate more with specific examples. ';
        if (q.type === 'behavioral' && !ans.answer.toLowerCase().match(/result|outcome|impact/)) feedback += 'Include specific outcomes and measurable results using the STAR method. ';
        if (q.type === 'technical' && wc < 40) feedback += 'Add more technical details, mention specific tools/algorithms/patterns. ';
        feedback += 'Review the model answer for this question to strengthen your response.';
        return { ...q, answer: ans.answer, score, feedback };
    });

    const totalScore = Math.round(results.reduce((s, r) => s + r.score, 0) / results.length);
    const answered = results.filter(r => !r.answer.includes('Skipped')).length;
    const goodAnswers = results.filter(r => r.score >= 70).length;

    // Animate score ring
    const ring = document.getElementById('mock-score-ring');
    const circumference = 327;
    setTimeout(() => { ring.style.strokeDashoffset = circumference - (circumference * totalScore / 100); }, 100);
    document.getElementById('mock-final-score').textContent = totalScore;

    document.getElementById('mock-score-breakdown').innerHTML = `
        <div class="score-metric"><div class="score-metric-label">Questions Answered</div><div class="score-metric-value">${answered}/${results.length}</div></div>
        <div class="score-metric"><div class="score-metric-label">Strong Answers</div><div class="score-metric-value">${goodAnswers}</div></div>
        <div class="score-metric"><div class="score-metric-label">Time Taken</div><div class="score-metric-value">${document.getElementById('mock-timer').textContent}</div></div>
        <div class="score-metric"><div class="score-metric-label">Avg Score</div><div class="score-metric-value">${totalScore}/100</div></div>
    `;

    document.getElementById('mock-feedback-list').innerHTML = results.map((r, i) => {
        const badge = r.score >= 70 ? 'good' : r.score >= 50 ? 'average' : 'poor';
        const badgeLabel = r.score >= 70 ? '✓ Strong' : r.score >= 50 ? '~ Average' : '✗ Weak';
        return `<div class="mock-feedback-item">
            <div class="question-label">Question ${i + 1} — ${r.type}</div>
            <div class="score-badge ${badge}">${badgeLabel} (${r.score}/100)</div>
            <div class="question-text">${r.q}</div>
            <div class="answer-text">"${r.answer}"</div>
            <div class="feedback-text">${r.feedback}</div>
        </div>`;
    }).join('');

    incrementStat('mocks');
    showToast('Mock interview completed! Review your performance.', 'success');
}

function resetMockInterview() {
    AppState.mockSession = null;
    clearInterval(AppState.mockTimer);
    document.getElementById('mock-setup').style.display = 'flex';
    document.getElementById('mock-session').style.display = 'none';
    document.getElementById('mock-results').style.display = 'none';
    // Reset score ring
    document.getElementById('mock-score-ring').style.strokeDashoffset = 327;
    
    stopMockMic();
    stopMockSpeaking();
}

// Profile button click
document.getElementById('user-profile-btn')?.addEventListener('click', showProfileModal);

// ==================== SUPABASE CONFIG ====================
const SUPABASE_URL = 'https://qzhodtpzajupwgoghmcw.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF6aG9kdHB6YWp1cHdnb2dobWN3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY0NDQ4NzksImV4cCI6MjA5MjAyMDg3OX0.av43zZ-nAZg4FOnHfXNRK_LKL-CAeUZR-ewgdn3VbsI';
let supabase;
try {
    supabase = window.supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
} catch (e) {
    console.warn('Supabase SDK not loaded, Skill Connect will work in offline mode.');
}

// ==================== AUTH (Email OTP) ====================
let authEmail = '';

async function checkAuthState() {
    if (!supabase) {
        // No Supabase — skip auth, show app directly
        showApp();
        return;
    }

    try {
        const { data: { session } } = await supabase.auth.getSession();
        if (session && session.user) {
            authEmail = session.user.email;
            showApp();
            updateUserDisplay(session.user.email);
        } else {
            showAuthScreen();
        }

        // Listen for auth state changes
        supabase.auth.onAuthStateChange((event, session) => {
            if (event === 'SIGNED_IN' && session) {
                authEmail = session.user.email;
                showApp();
                updateUserDisplay(session.user.email);
            } else if (event === 'SIGNED_OUT') {
                showAuthScreen();
            }
        });
    } catch (err) {
        console.error('Auth check error:', err);
        showApp(); // Fallback to showing app
    }
}

function showApp() {
    document.getElementById('auth-screen').style.display = 'none';
    document.getElementById('app-wrapper').classList.add('authenticated');
    document.getElementById('app-wrapper').style.display = '';
}

function showAuthScreen() {
    document.getElementById('auth-screen').style.display = 'flex';
    document.getElementById('app-wrapper').classList.remove('authenticated');
    document.getElementById('app-wrapper').style.display = 'none';
}

function updateUserDisplay(email) {
    const nameEl = document.getElementById('display-user-name');
    const roleEl = document.getElementById('display-user-role');
    if (nameEl) nameEl.textContent = email.split('@')[0];
    if (roleEl) roleEl.textContent = email;
}

async function sendOTP() {
    const emailInput = document.getElementById('auth-email');
    const email = emailInput.value.trim();

    if (!email) {
        showToast('Please enter your email address', 'error');
        return;
    }
    if (!email.includes('@')) {
        showToast('Please enter a valid email address', 'error');
        return;
    }

    authEmail = email;
    const btn = document.getElementById('auth-send-otp');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<div class="typing-indicator" style="justify-content:center"><span></span><span></span><span></span></div> Sending...`;

    try {
        const { error } = await supabase.auth.signInWithOtp({
            email: email,
            options: {
                shouldCreateUser: true,
            }
        });

        if (error) {
            showToast('Error: ' + error.message, 'error');
            btn.disabled = false;
            btn.innerHTML = originalText;
            return;
        }

        // Switch to OTP step
        document.getElementById('auth-step-email').style.display = 'none';
        document.getElementById('auth-step-otp').style.display = 'block';
        document.getElementById('auth-sent-email').textContent = email;

        // Focus first OTP box
        setTimeout(() => document.getElementById('otp-1').focus(), 100);

        showToast('OTP sent! Check your email inbox', 'success');

    } catch (err) {
        console.error('OTP error:', err);
        showToast('Network error. Please try again.', 'error');
    }

    btn.disabled = false;
    btn.innerHTML = originalText;
}

async function verifyOTP() {
    // Collect all 6 digits
    const otp = [1,2,3,4,5,6].map(i => document.getElementById(`otp-${i}`).value).join('');

    if (otp.length !== 6) {
        showToast('Please enter the complete 6-digit code', 'error');
        return;
    }

    const btn = document.getElementById('auth-verify-btn');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<div class="typing-indicator" style="justify-content:center"><span></span><span></span><span></span></div> Verifying...`;

    try {
        const { data, error } = await supabase.auth.verifyOtp({
            email: authEmail,
            token: otp,
            type: 'email',
        });

        if (error) {
            showToast('Invalid OTP: ' + error.message, 'error');
            btn.disabled = false;
            btn.innerHTML = originalText;
            // Clear OTP boxes
            [1,2,3,4,5,6].forEach(i => {
                const box = document.getElementById(`otp-${i}`);
                box.value = '';
                box.classList.remove('filled');
            });
            document.getElementById('otp-1').focus();
            return;
        }

        showToast('Welcome to Insignia! 🎉', 'success');
        // Auth state change listener will handle the rest

    } catch (err) {
        console.error('Verify error:', err);
        showToast('Network error. Please try again.', 'error');
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

async function resendOTP() {
    const btn = document.getElementById('auth-resend-btn');
    btn.disabled = true;
    btn.textContent = 'Sending...';

    try {
        const { error } = await supabase.auth.signInWithOtp({
            email: authEmail,
            options: { shouldCreateUser: true }
        });

        if (error) {
            showToast('Error: ' + error.message, 'error');
        } else {
            showToast('New OTP sent to ' + authEmail, 'success');
        }
    } catch (err) {
        showToast('Network error', 'error');
    }

    // Cooldown
    let cooldown = 30;
    btn.textContent = `Resend in ${cooldown}s`;
    const timer = setInterval(() => {
        cooldown--;
        btn.textContent = `Resend in ${cooldown}s`;
        if (cooldown <= 0) {
            clearInterval(timer);
            btn.disabled = false;
            btn.textContent = 'Resend OTP';
        }
    }, 1000);
}

function backToEmail() {
    document.getElementById('auth-step-email').style.display = 'block';
    document.getElementById('auth-step-otp').style.display = 'none';
    // Clear OTP boxes
    [1,2,3,4,5,6].forEach(i => {
        const box = document.getElementById(`otp-${i}`);
        box.value = '';
        box.classList.remove('filled');
    });
}

async function logoutUser() {
    if (!supabase) return;

    try {
        await supabase.auth.signOut();
        authEmail = '';
        localStorage.removeItem('sc_profile');
        showToast('Signed out successfully', 'info');
    } catch (err) {
        console.error('Logout error:', err);
    }
}

function setupOTPInputs() {
    const boxes = [1,2,3,4,5,6].map(i => document.getElementById(`otp-${i}`));
    if (!boxes[0]) return;

    boxes.forEach((box, idx) => {
        // Only allow digits
        box.addEventListener('input', (e) => {
            const val = e.target.value.replace(/[^0-9]/g, '');
            e.target.value = val;
            if (val) {
                e.target.classList.add('filled');
                // Auto-advance to next box
                if (idx < 5) boxes[idx + 1].focus();
            } else {
                e.target.classList.remove('filled');
            }

            // Auto-submit when all filled
            if (boxes.every(b => b.value.length === 1)) {
                setTimeout(() => verifyOTP(), 300);
            }
        });

        // Handle backspace
        box.addEventListener('keydown', (e) => {
            if (e.key === 'Backspace' && !box.value && idx > 0) {
                boxes[idx - 1].focus();
                boxes[idx - 1].value = '';
                boxes[idx - 1].classList.remove('filled');
            }
        });

        // Handle paste (full OTP paste)
        box.addEventListener('paste', (e) => {
            e.preventDefault();
            const pasted = (e.clipboardData.getData('text') || '').replace(/[^0-9]/g, '').slice(0, 6);
            if (pasted.length >= 6) {
                pasted.split('').forEach((digit, i) => {
                    if (boxes[i]) {
                        boxes[i].value = digit;
                        boxes[i].classList.add('filled');
                    }
                });
                boxes[5].focus();
                setTimeout(() => verifyOTP(), 300);
            }
        });

        // Allow Enter to submit
        box.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') verifyOTP();
        });
    });

    // Also allow Enter on email input
    const emailInput = document.getElementById('auth-email');
    if (emailInput) {
        emailInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendOTP();
        });
    }
}


// ==================== SKILL CONNECT ====================
const SKILL_SUGGESTIONS = [
    'Python', 'JavaScript', 'Java', 'C++', 'TypeScript', 'React', 'Angular', 'Vue.js',
    'Node.js', 'Django', 'Flask', 'Spring Boot', 'Express.js', 'Next.js', 'HTML/CSS',
    'SQL', 'MongoDB', 'PostgreSQL', 'Firebase', 'Supabase',
    'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision', 'TensorFlow', 'PyTorch',
    'Data Analysis', 'Pandas', 'NumPy', 'Power BI', 'Tableau', 'Excel',
    'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'CI/CD', 'DevOps',
    'Git', 'Linux', 'REST APIs', 'GraphQL', 'Microservices',
    'Flutter', 'React Native', 'Swift', 'Kotlin', 'Android', 'iOS',
    'Figma', 'UI/UX Design', 'Adobe XD', 'Photoshop',
    'Blockchain', 'Solidity', 'Web3', 'Cybersecurity', 'Ethical Hacking',
    'DSA', 'System Design', 'DBMS', 'Networking', 'Operating Systems',
    'Communication', 'Leadership', 'Problem Solving', 'Agile', 'Scrum'
];

// State for Skill Connect
const SCState = {
    teachSkills: [],
    learnSkills: [],
    profile: JSON.parse(localStorage.getItem('sc_profile') || 'null'),
    matches: [],
    currentFilter: 'all',
    mailTarget: null,
    loading: false,
};

// Init Skill Connect when page loads
window.addEventListener('DOMContentLoaded', () => {
    initSkillConnect();
});

function initSkillConnect() {
    // Setup tag input handlers
    setupTagInput('sc-teach-input', 'sc-teach-tags', 'sc-teach-suggestions', 'teach');
    setupTagInput('sc-learn-input', 'sc-learn-tags', 'sc-learn-suggestions', 'learn');

    // Load saved profile from localStorage first (fast)
    if (SCState.profile) {
        SCState.teachSkills = SCState.profile.teach || [];
        SCState.learnSkills = SCState.profile.learn || [];
        document.getElementById('sc-name').value = SCState.profile.name || '';
        document.getElementById('sc-email').value = SCState.profile.email || '';
        renderTags('sc-teach-tags', SCState.teachSkills, 'teach');
        renderTags('sc-learn-tags', SCState.learnSkills, 'learn');
        showMatchesView();
    }

    // Show suggestions
    showSuggestions('sc-teach-suggestions', 'teach');
    showSuggestions('sc-learn-suggestions', 'learn');
}

function setupTagInput(inputId, tagsContainerId, suggestionsId, type) {
    const input = document.getElementById(inputId);
    if (!input) return;

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ',') {
            e.preventDefault();
            const value = input.value.trim().replace(/,$/, '');
            if (value) {
                addSkillTag(value, type);
                input.value = '';
                showSuggestions(suggestionsId, type);
            }
        }
        if (e.key === 'Backspace' && !input.value) {
            const arr = type === 'teach' ? SCState.teachSkills : SCState.learnSkills;
            if (arr.length) {
                arr.pop();
                renderTags(tagsContainerId, arr, type);
                showSuggestions(suggestionsId, type);
            }
        }
    });

    input.addEventListener('input', () => {
        showSuggestions(suggestionsId, type, input.value);
    });
}

function addSkillTag(skill, type) {
    const normalizedSkill = skill.charAt(0).toUpperCase() + skill.slice(1);
    const arr = type === 'teach' ? SCState.teachSkills : SCState.learnSkills;
    if (arr.find(s => s.toLowerCase() === normalizedSkill.toLowerCase())) {
        showToast(`"${normalizedSkill}" is already added`, 'error');
        return;
    }
    arr.push(normalizedSkill);
    const containerId = type === 'teach' ? 'sc-teach-tags' : 'sc-learn-tags';
    renderTags(containerId, arr, type);
}

function removeSkillTag(skill, type) {
    const arr = type === 'teach' ? SCState.teachSkills : SCState.learnSkills;
    const idx = arr.findIndex(s => s.toLowerCase() === skill.toLowerCase());
    if (idx > -1) arr.splice(idx, 1);
    const containerId = type === 'teach' ? 'sc-teach-tags' : 'sc-learn-tags';
    const suggestionsId = type === 'teach' ? 'sc-teach-suggestions' : 'sc-learn-suggestions';
    renderTags(containerId, arr, type);
    showSuggestions(suggestionsId, type);
}

function renderTags(containerId, skills, type) {
    const container = document.getElementById(containerId);
    container.innerHTML = skills.map(s =>
        `<span class="sc-tag ${type}">
            ${s}
            <span class="sc-tag-remove" onclick="removeSkillTag('${s.replace(/'/g, "\\'")}', '${type}')">&times;</span>
        </span>`
    ).join('');
}

function showSuggestions(containerId, type, filter = '') {
    const container = document.getElementById(containerId);
    if (!container) return;
    const existing = type === 'teach' ? SCState.teachSkills : SCState.learnSkills;
    let available = SKILL_SUGGESTIONS.filter(s =>
        !existing.find(e => e.toLowerCase() === s.toLowerCase())
    );
    if (filter) {
        available = available.filter(s => s.toLowerCase().includes(filter.toLowerCase()));
    }
    const show = available.slice(0, 8);
    container.innerHTML = show.map(s =>
        `<span class="sc-suggestion-chip" onclick="addSuggestion('${s.replace(/'/g, "\\'")}', '${type}', '${containerId}')">${s}</span>`
    ).join('');
}

function addSuggestion(skill, type, suggestionsContainerId) {
    addSkillTag(skill, type);
    const inputId = type === 'teach' ? 'sc-teach-input' : 'sc-learn-input';
    document.getElementById(inputId).value = '';
    showSuggestions(suggestionsContainerId, type);
}

// ==================== SUPABASE SAVE & FETCH ====================
async function saveSkillProfile() {
    const name = document.getElementById('sc-name').value.trim();
    const email = document.getElementById('sc-email').value.trim();

    if (!name) { showToast('Please enter your name', 'error'); return; }
    if (!email) { showToast('Please enter your Gmail address', 'error'); return; }
    if (!email.includes('@')) { showToast('Please enter a valid email', 'error'); return; }
    if (SCState.teachSkills.length === 0 && SCState.learnSkills.length === 0) {
        showToast('Add at least one skill to teach or learn', 'error'); return;
    }

    const btn = document.getElementById('sc-save-btn');
    btn.classList.add('loading');

    if (!supabase) {
        // Offline fallback — save locally only
        SCState.profile = { name, email, teach: [...SCState.teachSkills], learn: [...SCState.learnSkills] };
        localStorage.setItem('sc_profile', JSON.stringify(SCState.profile));
        showToast('Profile saved locally (database unavailable)', 'info');
        btn.classList.remove('loading');
        setTimeout(() => { showMatchesView(); }, 400);
        return;
    }

    try {
        // Upsert profile to Supabase (insert or update based on email)
        const { data, error } = await supabase
            .from('skill_profiles')
            .upsert({
                name,
                email,
                teach_skills: [...SCState.teachSkills],
                learn_skills: [...SCState.learnSkills],
                updated_at: new Date().toISOString(),
            }, { onConflict: 'email' })
            .select();

        if (error) {
            console.error('Supabase error:', error);
            showToast('Error saving profile: ' + error.message, 'error');
            btn.classList.remove('loading');
            return;
        }

        // Save locally too for fast loading
        SCState.profile = {
            name,
            email,
            teach: [...SCState.teachSkills],
            learn: [...SCState.learnSkills],
        };
        localStorage.setItem('sc_profile', JSON.stringify(SCState.profile));

        showToast('Profile saved to database! Finding matches...', 'success');

        btn.classList.remove('loading');
        setTimeout(() => { showMatchesView(); }, 400);

    } catch (err) {
        console.error('Save error:', err);
        showToast('Network error. Please try again.', 'error');
        btn.classList.remove('loading');
    }
}

async function showMatchesView() {
    document.getElementById('sc-setup').style.display = 'none';
    document.getElementById('sc-matches-section').style.display = 'block';

    renderProfileBanner();
    await computeMatches();
    renderMatches();
}

function editSkillProfile() {
    document.getElementById('sc-setup').style.display = '';
    document.getElementById('sc-matches-section').style.display = 'none';

    // Re-render tags and suggestions
    renderTags('sc-teach-tags', SCState.teachSkills, 'teach');
    renderTags('sc-learn-tags', SCState.learnSkills, 'learn');
    showSuggestions('sc-teach-suggestions', 'teach');
    showSuggestions('sc-learn-suggestions', 'learn');
}

function renderProfileBanner() {
    const p = SCState.profile;
    const initials = p.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
    const allSkills = [...p.teach.map(s => ({ s, type: 'teach' })), ...p.learn.map(s => ({ s, type: 'learn' }))];

    document.getElementById('sc-profile-banner').innerHTML = `
        <div class="sc-banner-avatar">${initials}</div>
        <div class="sc-banner-info">
            <div class="sc-banner-name">${p.name}</div>
            <div class="sc-banner-email">${p.email}</div>
            <div class="sc-banner-skills">
                ${allSkills.map(({ s, type }) => `<span class="sc-tag ${type}">${s}</span>`).join('')}
            </div>
        </div>
        <div class="sc-banner-stats">
            <div>
                <div class="sc-banner-stat-val">${p.teach.length}</div>
                <div class="sc-banner-stat-label">Can Teach</div>
            </div>
            <div>
                <div class="sc-banner-stat-val">${p.learn.length}</div>
                <div class="sc-banner-stat-label">Want to Learn</div>
            </div>
            <div>
                <div class="sc-banner-stat-val" id="sc-match-count">${SCState.matches.length || '—'}</div>
                <div class="sc-banner-stat-label">Matches</div>
            </div>
        </div>
    `;
}

async function computeMatches() {
    const me = SCState.profile;
    SCState.matches = [];

    // Show loading state in grid
    const grid = document.getElementById('sc-matches-grid');
    grid.innerHTML = `<div class="sc-empty">
        <div class="typing-indicator"><span></span><span></span><span></span></div>
        <h3>Finding matches from database...</h3>
    </div>`;

    let users = [];
    const mockUsers = [
        { name: "Alex Chen", email: "alex.c@example.com", teach_skills: ["React", "JavaScript", "HTML/CSS"], learn_skills: ["Python", "Node.js"] },
        { name: "Sarah J.", email: "sarah@example.com", teach_skills: ["Python", "Data Analysis", "Machine Learning"], learn_skills: ["AWS", "Docker"] },
        { name: "David K.", email: "david@example.com", teach_skills: ["AWS", "System Design", "Docker"], learn_skills: ["React", "JavaScript"] },
        { name: "Elena R.", email: "elena@example.com", teach_skills: ["UI/UX Design", "HTML/CSS", "Figma"], learn_skills: ["JavaScript", "React"] }
    ];

    if (!supabase) {
        users = mockUsers;
        showToast('Offline mode: Using sample profiles', 'info');
    } else {
        try {
            const { data, error } = await supabase
                .from('skill_profiles')
                .select('*')
                .neq('email', me.email);

            if (error || !data || data.length === 0) {
                if (error) console.error('Fetch error:', error);
                users = mockUsers;
            } else {
                users = data;
            }
        } catch (err) {
            console.error('Match error:', err);
            users = mockUsers;
        }
    }

    try {
        // Compute matches against real DB users
        users.forEach(user => {
            const userTeach = user.teach_skills || [];
            const userLearn = user.learn_skills || [];

            // Skills they teach that I want to learn
            const canTeachMe = userTeach.filter(s =>
                me.learn.find(l => l.toLowerCase() === s.toLowerCase())
            );
            // Skills I teach that they want to learn
            const wantsFromMe = userLearn.filter(s =>
                me.teach.find(t => t.toLowerCase() === s.toLowerCase())
            );

            if (canTeachMe.length === 0 && wantsFromMe.length === 0) return;

            const matchScore = (canTeachMe.length + wantsFromMe.length) * 100 /
                Math.max(1, me.learn.length + me.teach.length);
            const score = Math.min(99, Math.round(matchScore * 2.5 + 20));

            let reason = '';
            if (canTeachMe.length > 0 && wantsFromMe.length > 0) {
                reason = `${user.name.split(' ')[0]} can teach you ${canTeachMe.join(', ')} and wants to learn ${wantsFromMe.join(', ')} from you — a perfect mutual exchange!`;
            } else if (canTeachMe.length > 0) {
                reason = `${user.name.split(' ')[0]} is proficient in ${canTeachMe.join(', ')} which you want to learn.`;
            } else {
                reason = `${user.name.split(' ')[0]} wants to learn ${wantsFromMe.join(', ')} — something you can teach!`;
            }

            SCState.matches.push({
                name: user.name,
                email: user.email,
                teach: userTeach,
                learn: userLearn,
                canTeachMe,
                wantsFromMe,
                score,
                reason,
                matchType: canTeachMe.length > 0 && wantsFromMe.length > 0 ? 'mutual' :
                            canTeachMe.length > 0 ? 'canTeach' : 'wantsToLearn',
            });
        });

        // Sort by score descending
        SCState.matches.sort((a, b) => b.score - a.score);

        // Update banner count
        const countEl = document.getElementById('sc-match-count');
        if (countEl) countEl.textContent = SCState.matches.length;

    } catch (err) {
        console.error('Match computation error:', err);
        showToast('Error finding matches. Please try again.', 'error');
    }
}

function getAvatarGradient(index) {
    const gradients = [
        'linear-gradient(135deg, #6C5CE7, #a29bfe)',
        'linear-gradient(135deg, #00B4DB, #00D2FF)',
        'linear-gradient(135deg, #FD79A8, #fdcbdd)',
        'linear-gradient(135deg, #FDCB6E, #f9e7a1)',
        'linear-gradient(135deg, #00B894, #55efc4)',
        'linear-gradient(135deg, #e17055, #fab1a0)',
        'linear-gradient(135deg, #0984e3, #74b9ff)',
        'linear-gradient(135deg, #6c5ce7, #fd79a8)',
    ];
    return gradients[index % gradients.length];
}

function filterMatches(filter, btn) {
    SCState.currentFilter = filter;
    document.querySelectorAll('.sc-tab').forEach(t => t.classList.remove('active'));
    if (btn) btn.classList.add('active');
    renderMatches();
}

function renderMatches() {
    const container = document.getElementById('sc-matches-grid');
    let filtered = SCState.matches;

    if (SCState.currentFilter === 'canTeach') {
        filtered = filtered.filter(m => m.canTeachMe.length > 0);
    } else if (SCState.currentFilter === 'wantsToLearn') {
        filtered = filtered.filter(m => m.wantsFromMe.length > 0);
    }

    if (filtered.length === 0) {
        container.innerHTML = `
            <div class="sc-empty">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>
                    <path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>
                </svg>
                <h3>No matches found</h3>
                <p>Try adding more skills to teach or learn to discover connections.</p>
            </div>
        `;
        return;
    }

    container.innerHTML = filtered.map((m, i) => {
        const initials = m.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
        const scoreClass = m.score >= 70 ? 'high' : m.score >= 45 ? 'medium' : 'low';
        const gradient = getAvatarGradient(i);

        return `
            <div class="sc-match-card" style="animation-delay: ${i * 0.08}s">
                <div class="sc-match-header">
                    <div class="sc-match-avatar" style="background: ${gradient}">${initials}</div>
                    <div>
                        <div class="sc-match-name">${m.name}</div>
                        <div class="sc-match-email">${m.email}</div>
                    </div>
                    <div class="sc-match-score ${scoreClass}">${m.score}%</div>
                </div>

                <div class="sc-match-reason">${m.reason}</div>

                ${m.canTeachMe.length > 0 ? `
                    <div class="sc-match-section">
                        <div class="sc-match-label">Can Teach You</div>
                        <div class="sc-match-tags">
                            ${m.canTeachMe.map(s => `<span class="sc-match-tag highlight">★ ${s}</span>`).join('')}
                            ${m.teach.filter(s => !m.canTeachMe.find(c => c.toLowerCase() === s.toLowerCase())).map(s => `<span class="sc-match-tag normal">${s}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}

                ${m.wantsFromMe.length > 0 ? `
                    <div class="sc-match-section">
                        <div class="sc-match-label">Wants to Learn From You</div>
                        <div class="sc-match-tags">
                            ${m.wantsFromMe.map(s => `<span class="sc-match-tag want">✦ ${s}</span>`).join('')}
                            ${m.learn.filter(s => !m.wantsFromMe.find(c => c.toLowerCase() === s.toLowerCase())).map(s => `<span class="sc-match-tag normal">${s}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}

                <div class="sc-match-footer">
                    <button class="btn btn-outline btn-sm" onclick="openMailModal(${i})">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/></svg>
                        Message via Gmail
                    </button>
                </div>
            </div>
        `;
    }).join('');
}

// ==================== GMAIL MESSAGING ====================
function openMailModal(matchIndex) {
    const filtered = getFilteredMatches();
    const match = filtered[matchIndex];
    if (!match) return;
    SCState.mailTarget = match;

    const initials = match.name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);

    document.getElementById('sc-mail-title').textContent = `Message ${match.name}`;
    document.getElementById('sc-mail-recipient').innerHTML = `
        <div class="sc-mail-recipient-avatar">${initials}</div>
        <div class="sc-mail-recipient-info">
            <div class="sc-mail-recipient-name">${match.name}</div>
            <div class="sc-mail-recipient-email">${match.email}</div>
        </div>
    `;

    // Pre-fill subject and body
    const myName = SCState.profile.name;
    let subject = '';
    let body = '';

    if (match.canTeachMe.length > 0 && match.wantsFromMe.length > 0) {
        subject = `Skill Exchange: ${match.canTeachMe[0]} ↔ ${match.wantsFromMe[0]}`;
        body = `Hi ${match.name.split(' ')[0]},\n\nI found your profile on Insignia Skill Connect! I noticed you're skilled in ${match.canTeachMe.join(', ')} — which is exactly what I'm looking to learn.\n\nIn return, I can help you with ${match.wantsFromMe.join(', ')} since I have experience in those areas.\n\nWould you be open to a skill exchange session? We could set up a time that works for both of us.\n\nLooking forward to hearing from you!\n\nBest regards,\n${myName}`;
    } else if (match.canTeachMe.length > 0) {
        subject = `Learning ${match.canTeachMe[0]} — From Insignia Skill Connect`;
        body = `Hi ${match.name.split(' ')[0]},\n\nI found your profile on Insignia Skill Connect and I'm impressed by your expertise in ${match.canTeachMe.join(', ')}.\n\nI'm currently looking to learn ${match.canTeachMe.join(' and ')} and would love the opportunity to learn from you. Would you be open to a mentoring session or a quick chat?\n\nThanks!\n\nBest regards,\n${myName}`;
    } else {
        subject = `I Can Help You Learn ${match.wantsFromMe[0]} — Insignia Skill Connect`;
        body = `Hi ${match.name.split(' ')[0]},\n\nI came across your profile on Insignia Skill Connect and noticed you're interested in learning ${match.wantsFromMe.join(', ')}.\n\nI have experience in those areas and would be happy to help! Whether it's a quick walkthrough, pair programming, or sharing resources — I'm open to however you'd like to learn.\n\nLet me know if you're interested!\n\nBest regards,\n${myName}`;
    }

    document.getElementById('sc-mail-subject').value = subject;
    document.getElementById('sc-mail-body').value = body;

    document.getElementById('sc-mail-modal').style.display = 'flex';
}

function closeMailModal() {
    document.getElementById('sc-mail-modal').style.display = 'none';
    SCState.mailTarget = null;
}

function sendViaGmail() {
    const target = SCState.mailTarget;
    if (!target) return;

    const subject = document.getElementById('sc-mail-subject').value;
    const body = document.getElementById('sc-mail-body').value;

    // Open Gmail compose in a new tab
    const gmailUrl = `https://mail.google.com/mail/?view=cm&fs=1&to=${encodeURIComponent(target.email)}&su=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;

    window.open(gmailUrl, '_blank');

    closeMailModal();
    showToast(`Gmail opened to message ${target.name}!`, 'success');
}

function getFilteredMatches() {
    let filtered = SCState.matches;
    if (SCState.currentFilter === 'canTeach') {
        filtered = filtered.filter(m => m.canTeachMe.length > 0);
    } else if (SCState.currentFilter === 'wantsToLearn') {
        filtered = filtered.filter(m => m.wantsFromMe.length > 0);
    }
    return filtered;
}
