import streamlit as st
from jinja2 import Template

# ------------------- COVER LETTER GENERATOR -------------------

def generate_cover_letter(tailored_resume, job_text, tone="professional"):
    """
    Generate a cover letter based on the tailored resume and job description.
    
    Args:
        tailored_resume (str): The tailored resume text.
        job_text (str): The job description text.
        tone (str): Tone of the cover letter ('professional', 'creative', 'concise').
        
    Returns:
        str: The generated cover letter text.
    """
    template = Template("""
Dear {{ hiring_manager_name }},

I am writing to express my interest in the {{ position }} role at {{ company_name }}. With {{ years_of_experience }} years of experience in {{ relevant_industries }}, I am confident that my background and skills make me an ideal candidate for this position.

{{ tailored_resume }}

In addition to my technical expertise, I bring a strong track record of {{ achievements }}. My ability to {{ key_skills }} has enabled me to successfully {{ recent_projects }} at previous companies.

I am particularly drawn to {{ company_name }} because of its commitment to {{ company_values }} and its innovative approach to {{ industry }}. I am eager to contribute my skills and experience to your team and help drive the success of this organization.

Thank you for considering my application. I look forward to the opportunity to discuss how my background, skills, and enthusiasms align with the needs of {{ company_name }}.

Sincerely,
{{ applicant_name }}
""")
    
    # Example data - replace with dynamic data from user input or session state
    data = {
        "hiring_manager_name": "John Doe",
        "position": "Software Engineer",
        "company_name": "Tech Innovations Inc.",
        "years_of_experience": 5,
        "relevant_industries": "software development, machine learning",
        "achievements": "developed and deployed scalable web applications, improved system performance by 20%",
        "key_skills": "problem-solving, teamwork, communication",
        "recent_projects": "led a team to develop a new AI-powered chatbot, resulting in a 15% increase in customer satisfaction",
        "company_values": "innovation, collaboration, excellence",
        "industry": "technology",
        "applicant_name": "Jane Smith"
    }
    
    return template.render(data)

# ------------------- Streamlit Component -------------------

def cover_letter_generator_component(tailored_resume, job_text):
    st.markdown('<p class="section-header">Cover Letter Generator</p>', unsafe_allow_html=True)
    
    tone = st.radio(
        "Tone",
        ["Professional", "Creative", "Concise"],
        help="Select the tone for your cover letter"
    )
    
    if st.button("GENERATE COVER LETTER"):
        cover_letter = generate_cover_letter(tailored_resume, job_text, tone.lower())
        
        with st.expander("Preview Cover Letter", expanded=True):
            st.markdown(f'<div class="preview-box">{cover_letter}</div>', unsafe_allow_html=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Cover_Letter_{timestamp}.txt"
        
        with open(filename, "w") as f:
            f.write(cover_letter)
        
        st.download_button(
            "DOWNLOAD COVER LETTER",
            data=cover_letter,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )

# ------------------- Main Application Integration -------------------

def main():
    # ... existing code ...

    if uploaded_file and job_input:
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        
        with col_btn2:
            generate_btn = st.button(
                "GENERATE TAILORED RESUME",
                use_container_width=True,
                disabled=not st.session_state.connected
            )
        
        if not st.session_state.connected:
            st.warning("Connect to an LLM in the sidebar to continue")
        
        if generate_btn and st.session_state.connected:
            resume_text = extract_resume_text(uploaded_file)
            keywords = extract_keywords(job_input)
            
            initial_score = calculate_match_score(resume_text, job_input)
            
            with st.spinner("Processing..."):
                tailored_resume = generate_tailored_resume(resume_text, job_input, keywords)
        
            if tailored_resume:
                new_score = calculate_match_score(tailored_resume, job_input)
                improvement = new_score - initial_score
            
                st.success("Generation complete")
                
                # Score display
                st.markdown(f"""
                <div class="score-container">
                    <div class="score-box">
                        <div class="score-label">Original Score</div>
                        <div class="score-value original">{initial_score}%</div>
                    </div>
                    <div class="score-box">
                        <div class="score-label">Improvement</div>
                        <div class="score-value improvement">+{improvement:.1f}%</div>
                    </div>
                    <div class="score-box">
                        <div class="score-label">Optimized Score</div>
                        <div class="score-value">{new_score}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # ------------------- SKILL GAP ANALYZER -------------------
                st.markdown('<p class="section-header">Skill Gap Analysis</p>', unsafe_allow_html=True)
                
                with st.spinner("Analyzing skill gaps..."):
                    gap_results = analyze_skill_gaps(resume_text, job_input, nlp)
            
                # Summary stats
                st.markdown(f"""
                <div class="analysis-card">
                    <div class="stat-row">
                        <span class="stat-label">Skill Coverage</span>
                        <span class="stat-value">{gap_results['coverage_score']}%</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Matching Skills</span>
                        <span class="stat-value">{len(gap_results['matching_skills'])}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Missing Skills</span>
                        <span class="stat-value">{len(gap_results['missing_skills'])}</span>
                    </div>
                    <div class="stat-row" style="border-bottom: none;">
                        <span class="stat-label">Additional Skills</span>
                        <span class="stat-value">{len(gap_results['extra_skills'])}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Skills visualization
                with st.expander("Skills Breakdown", expanded=True):
                    skill_col1, skill_col2 = st.columns(2)
                    
                    with skill_col1:
                        st.markdown("**Matching Skills**")
                        if gap_results['matching_skills']:
                            skills_html = " ".join([f'<span class="skill-tag skill-match">{skill}</span>' for skill in sorted(gap_results['matching_skills'])])
                            st.markdown(skills_html, unsafe_allow_html=True)
                        else:
                            st.text("No matching skills found")
                
                    with skill_col2:
                        st.markdown("**Your Additional Skills**")
                        if gap_results['extra_skills']:
                            skills_html = " ".join([f'<span class="skill-tag skill-extra">{skill}</span>' for skill in sorted(gap_results['extra_skills'])])
                            st.markdown(skills_html, unsafe_allow_html=True)
                        else:
                            st.text("None")
            
                # Gap analysis table
                if gap_results['gap_analysis']:
                    with st.expander("Missing Skills - Action Plan", expanded=True):
                        st.markdown("**Skills to Add to Your Resume:**")
                        
                        # Build table HTML
                        table_html = """
                        <table class="gap-table">
                            <thead>
                                <tr>
                                    <th>Skill</th>
                                    <th>Priority</th>
                                    <th>Category</th>
                                    <th>Suggested Bullet Point</th>
                                    <th>Learn</th>
                                </tr>
                            </thead>
                            <tbody>
                        """
                        
                        for gap in gap_results['gap_analysis']:
                            priority_class = f"priority-{gap['priority']}"
                            table_html += f"""
                            <tr>
                                <td><span class="skill-tag skill-gap">{gap['skill']}</span></td>
                                <td><span class="{priority_class}">{gap['priority'].upper()}</span></td>
                                <td>{gap['category'].replace('_', ' ').title()}</td>
                                <td>{gap['suggestion']}</td>
                                <td><a href="{gap['learning_url']}" target="_blank" class="learning-link">LinkedIn Learning</a></td>
                            </tr>
                            """
                        
                        table_html += "</tbody></table>"
                        st.markdown(table_html, unsafe_allow_html=True)
                        
                        st.markdown("---")
                        st.markdown("**Priority Legend:** ", unsafe_allow_html=True)
                        st.markdown('<span class="priority-high">HIGH</span> = Mentioned 3+ times or in requirements | <span class="priority-medium">MEDIUM</span> = Mentioned 2 times or preferred | <span class="priority-low">LOW</span> = Nice to have', unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Preview and download
                st.markdown('<p class="section-header">Output: Tailored Resume</p>', unsafe_allow_html=True)
                st.markdown(f'<div class="preview-box">{tailored_resume}</div>', unsafe_allow_html=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"Tailored_Resume_{timestamp}.docx"
                filepath = create_professional_docx(tailored_resume, filename)
                
                st.markdown("---")
                
                dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
                with dl_col2:
                    with open(filepath, "rb") as f:
                        st.download_button(
                            "DOWNLOAD DOCX",
                            data=f,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )
                
                # Cover Letter Generator Component
                cover_letter_generator_component(tailored_resume, job_input)

    else:
        st.info("Upload a resume and provide a job description to begin")

if __name__ == "__main__":
    main()