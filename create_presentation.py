"""
Script to generate the comprehensive, professional academic presentation for the
ICU Bed Allocation & Patient Prioritization Reinforcement Learning Project.
Outputs: ICU_Bed_Allocation_Presentation.pptx
"""
import sys
import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR
FIG_DIR = SCRIPT_DIR / "figures"
OUTPUT_PATH = SCRIPT_DIR / "ICU_Bed_Allocation_Presentation.pptx"
ROOT_OUTPUT_PATH = SCRIPT_DIR.parent / "ICU_Bed_Allocation_Presentation.pptx"

# Color Palette (Professional Clinical & Academic Theme)
NAVY = RGBColor(15, 30, 54)        # #0F1E36 - Deep Navy Blue (Primary Header / Title)
TEAL = RGBColor(0, 150, 136)       # #009688 - Medical Cyan / Teal (Accent / Brand)
DARK_TEAL = RGBColor(0, 105, 95)   # #00695F - Darker Teal
CORAL = RGBColor(217, 83, 79)      # #D9534F - Critical Severity Alert (High SOFA)
BG_LIGHT = RGBColor(248, 250, 252) # #F8FAFC - Clean Slate White
WHITE = RGBColor(255, 255, 255)    # #FFFFFF
TEXT_DARK = RGBColor(30, 41, 59)   # #1E293B - Slate 800 (Primary Body Text)
TEXT_MUTED = RGBColor(100, 116, 139) # #64748B - Slate 500
BORDER_COL = RGBColor(226, 232, 240) # #E2E8F0 - Soft Card Border
CARD_BG = RGBColor(255, 255, 255)
CARD_BG_TINT = RGBColor(241, 245, 249) # #F1F5F9 - Light Blue-Grey Card
HEADER_BLUE = RGBColor(26, 54, 93) # #1A365D - Slide Header Banner

def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # Blank slide

    def add_header(slide, title_text, category_text="ICU BED ALLOCATION PROJECT | IEEE RL STUDY"):
        # Header banner shape
        header_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        header_shape.fill.solid()
        header_shape.fill.fore_color.rgb = NAVY
        header_shape.line.fill.background()

        # Category / Tagline
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.12), Inches(11.5), Inches(0.28))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_top = tf_cat.margin_right = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(9.5)
        p_cat.font.bold = True
        p_cat.font.color.rgb = TEAL

        # Slide Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.40), Inches(11.5), Inches(0.60))
        tf_t = title_box.text_frame
        tf_t.word_wrap = True
        tf_t.margin_left = tf_t.margin_top = tf_t.margin_right = tf_t.margin_bottom = 0
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(22)
        p_t.font.bold = True
        p_t.font.color.rgb = WHITE

        # Footer band
        footer_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.15), Inches(13.333), Inches(0.35))
        footer_shape.fill.solid()
        footer_shape.fill.fore_color.rgb = BG_LIGHT
        footer_shape.line.color.rgb = BORDER_COL

        foot_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.20), Inches(11.733), Inches(0.25))
        tf_f = foot_box.text_frame
        tf_f.margin_left = tf_f.margin_top = tf_f.margin_right = tf_f.margin_bottom = 0
        p_f = tf_f.paragraphs[0]
        p_f.text = "Reinforcement Learning for ICU Bed Allocation & Patient Prioritization | Event-Driven SMDP Framework"
        p_f.font.size = Pt(8.5)
        p_f.font.color.rgb = TEXT_MUTED

    def add_card(slide, left, top, width, height, bg_color=CARD_BG, border_color=BORDER_COL):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1)
        else:
            shape.line.fill.background()
        return shape

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Dark Elegance)
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    # Accent decorative bar
    bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.3), Inches(1.5), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()

    # Title & Subtitle box
    tbox = slide1.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.333), Inches(3.2))
    tf1 = tbox.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "RESEARCH PROJECT PRESENTATION"
    p0.font.size = Pt(12)
    p0.font.bold = True
    p0.font.color.rgb = TEAL
    p0.space_after = Pt(14)

    p1 = tf1.add_paragraph()
    p1.text = "Reinforcement Learning for ICU Bed Allocation\nand Patient Prioritization"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_after = Pt(16)

    p2 = tf1.add_paragraph()
    p2.text = "An Event-Driven Semi-Markov Decision Process (SMDP) under Constrained Capacity & Diurnal Arrival Windows"
    p2.font.size = Pt(15)
    p2.font.color.rgb = RGBColor(203, 213, 225)

    # Key Highlights Badges
    badges = [
        ("Capacity: C = 20 Beds", "Single Hospital Unit"),
        ("48 Diurnal Windows", "Full-Day 30-min Cohorts"),
        ("Vincent 1996 SOFA", "Computed from 18 Raw Labs"),
        ("Double DQN Selected", "High Seed Robustness")
    ]
    for i, (b_title, b_sub) in enumerate(badges):
        bx = Inches(1.0 + i * 2.85)
        by = Inches(5.1)
        b_shape = slide1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, by, Inches(2.7), Inches(1.2))
        b_shape.fill.solid()
        b_shape.fill.fore_color.rgb = RGBColor(26, 44, 76)
        b_shape.line.color.rgb = RGBColor(51, 75, 115)

        tb_b = slide1.shapes.add_textbox(bx + Inches(0.15), by + Inches(0.15), Inches(2.4), Inches(0.9))
        tf_b = tb_b.text_frame
        tf_b.word_wrap = True
        pb1 = tf_b.paragraphs[0]
        pb1.text = b_title
        pb1.font.size = Pt(13)
        pb1.font.bold = True
        pb1.font.color.rgb = TEAL
        pb2 = tf_b.add_paragraph()
        pb2.text = b_sub
        pb2.font.size = Pt(10)
        pb2.font.color.rgb = RGBColor(148, 163, 184)

    # Footer author / course tag
    f_tag = slide1.shapes.add_textbox(Inches(1.0), Inches(6.6), Inches(11.333), Inches(0.4))
    tf_tag = f_tag.text_frame
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = "Department of Computer Science & Engineering | Reinforcement Learning Project Defense"
    p_tag.font.size = Pt(10)
    p_tag.font.color.rgb = RGBColor(148, 163, 184)

    # Helper function for adding structured slide content with 2 cards
    def add_two_column_slide(title, cat, left_title, left_bullets, right_title, right_bullets, left_w=5.7, right_w=5.7):
        slide = prs.slides.add_slide(blank_layout)
        add_header(slide, title, cat)

        # Left Card
        add_card(slide, Inches(0.8), Inches(1.4), Inches(left_w), Inches(5.5))
        ltb = slide.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(left_w - 0.4), Inches(5.1))
        ltf = ltb.text_frame
        ltf.word_wrap = True
        lp0 = ltf.paragraphs[0]
        lp0.text = left_title
        lp0.font.size = Pt(16)
        lp0.font.bold = True
        lp0.font.color.rgb = NAVY
        lp0.space_after = Pt(12)

        for b in left_bullets:
            p = ltf.add_paragraph()
            p.text = "• " + b[0]
            p.font.size = Pt(12.5)
            p.font.bold = True
            p.font.color.rgb = TEXT_DARK
            if len(b) > 1 and b[1]:
                p_sub = ltf.add_paragraph()
                p_sub.text = "   " + b[1]
                p_sub.font.size = Pt(11)
                p_sub.font.color.rgb = TEXT_MUTED
                p_sub.space_after = Pt(8)
            else:
                p.space_after = Pt(8)

        # Right Card
        rx = Inches(0.8 + left_w + 0.3)
        add_card(slide, rx, Inches(1.4), Inches(right_w), Inches(5.5))
        rtb = slide.shapes.add_textbox(rx + Inches(0.2), Inches(1.55), Inches(right_w - 0.4), Inches(5.1))
        rtf = rtb.text_frame
        rtf.word_wrap = True
        rp0 = rtf.paragraphs[0]
        rp0.text = right_title
        rp0.font.size = Pt(16)
        rp0.font.bold = True
        rp0.font.color.rgb = NAVY
        rp0.space_after = Pt(12)

        for b in right_bullets:
            p = rtf.add_paragraph()
            p.text = "• " + b[0]
            p.font.size = Pt(12.5)
            p.font.bold = True
            p.font.color.rgb = TEXT_DARK
            if len(b) > 1 and b[1]:
                p_sub = rtf.add_paragraph()
                p_sub.text = "   " + b[1]
                p_sub.font.size = Pt(11)
                p_sub.font.color.rgb = TEXT_MUTED
                p_sub.space_after = Pt(8)
            else:
                p.space_after = Pt(8)

        return slide

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # =========================================================================
    add_two_column_slide(
        title="Clinical Problem & Operational Motivation",
        cat="Section 1 | Project Overview & Challenges",
        left_title="The Critical Care Challenge in ICUs",
        left_bullets=[
            ("Strict Bed Capacity Constraint", "ICUs have a rigid physical limit (C = 20 beds in our single-hospital model). Staff, equipment, and infection control prevent ad-hoc bed expansion."),
            ("Stochastic Arrival Demand", "Emergency patients arrive at random hours with high time-of-day demand swings (peaks in afternoon, troughs at night)."),
            ("Uncertain Patient Stay Durations", "Actual length of stay (LOS) is unknown at presentation, depending on complex recovery and latent biological severity."),
            ("Waiting-Time Clinical Deterioration", "Patients left in the queue face dynamic hazard rates of sudden decompensation (septic shock, respiratory collapse).")
        ],
        right_title="Why Traditional Allocation Rules Fail",
        right_bullets=[
            ("First-Come, First-Served (FCFS)", "Blind to medical acuity. A stable patient arriving 2 minutes earlier blocks an acutely septic patient arriving later."),
            ("Highest-SOFA-First (Strict Triage)", "Admits the sickest patients first, but these patients occupy beds for weeks, completely blocking the ICU and causing dozens of moderate patients to deteriorate while waiting."),
            ("Shortest-Job-First (SPF Heuristic)", "Minimizes queue lengths by prioritizing quick stays, but severely ill, long-stay patients are indefinitely starved of care."),
            ("The Goal of Reinforcement Learning", "Learn an adaptive, dynamic policy that intelligently balances queue throughput with ethical patient acuity.")
        ]
    )

    # =========================================================================
    # SLIDE 3: KEY CONTRIBUTIONS & NOVELTIES
    # =========================================================================
    add_two_column_slide(
        title="Key Architectural & Methodological Contributions",
        cat="Section 1 | Research Scope & Novelty",
        left_title="Core Methodological Innovations",
        left_bullets=[
            ("Event-Driven Semi-Markov Formulation", "Transitions occur strictly at asynchronous clinical events (admissions, discharges, deteriorations), not arbitrary artificial clock ticks."),
            ("Continuous-Time Discounting", "Applies discount factor gamma^(dt / tau0) where gamma = 0.99 and tau0 = 1.0 h. Zero discounting during intra-epoch multi-admissions (dt = 0)."),
            ("48 Diurnal 30-Minute Windows", "Simulates realistic full-day hospital operational dynamics: 48 daily time slots (00:00 to 23:30) with sinusoidal arrival modulation."),
            ("Length-Biased Initial ICU Census", "Accurately models existing bed occupants (mean ~18/20 occupied) who release beds according to hidden remaining stay distributions.")
        ],
        right_title="Engineering & Algorithmic Highlights",
        right_bullets=[
            ("Permutation-Equivariant SetNet", "Deep Sets neural architecture that processes variable-length patient queues without arbitrary row-ordering biases."),
            ("Dual Objective Rewards with Exact Integration", "Reward 1 optimizes aggregate waiting; Reward 2 incorporates severity-weighting via exact closed-form integral calculus."),
            ("Strict Leakage Prevention Protocol", "Clean physical separation of observable vs hidden data; 21 automated leakage tests verify no ground-truth peeking."),
            ("Rigorous Multi-Seed Benchmarking", "5 seeds x 6,000 training episodes evaluated on 200 held-out test episodes with 4,000-sample bootstrap confidence intervals.")
        ]
    )

    # =========================================================================
    # SLIDE 4: CLINICAL DATA ENGINE & SYNTHETIC COHORT GENERATION
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Clinical Data Engine & Patient Generation", "Section 2.1 | Data Methodology")

    add_card(slide4, Inches(0.8), Inches(1.4), Inches(6.0), Inches(5.5))
    tb4 = slide4.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.6), Inches(5.1))
    tf4 = tb4.text_frame
    tf4.word_wrap = True
    p = tf4.paragraphs[0]
    p.text = "Biological Generative Process"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(10)

    bullets4 = [
        ("Latent Severity z ~ Beta(2, 3)", "Continuous severity in [0, 1] driving all biological manifestations. Census occupants are scaled by 0.85 (stabilized)."),
        ("18 Raw Physiological Measurements", "PaO2, FiO2, PF ratio, ventilation, platelets, bilirubin, MAP, vasopressors (dopamine/norepi), GCS, creatinine, urine output, vitals."),
        ("Patient-Specific Biological Noise nz ~ N(0, I)", "14-dim fixed noise vector ensures that when a patient deteriorates (z -> z + dz), all 18 measurements update in biological harmony."),
        ("Hidden vs. Observable Data Partitioning", "Observable: snapshots, elapsed census times, predictions.\nHidden: actual LOS, true hazard, frailty, latent z. Zero leakage confirmed across 21 test checks."),
        ("Total Scale of Dataset", "1,000 episodes (700 train / 100 val / 200 test), 26,882 patients (8,881 cohort arrivals, 18,001 census occupants).")
    ]
    for b in bullets4:
        p1 = tf4.add_paragraph()
        p1.text = "• " + b[0]
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_DARK
        p2 = tf4.add_paragraph()
        p2.text = "   " + b[1]
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_after = Pt(6)

    # Right Image Card: dataset_overview.png or raw_clinical_distributions.png
    add_card(slide4, Inches(7.1), Inches(1.4), Inches(5.4), Inches(5.5))
    fig_path = FIG_DIR / "raw_clinical_distributions.png"
    if fig_path.exists():
        slide4.shapes.add_picture(str(fig_path), Inches(7.25), Inches(1.55), width=Inches(5.1))
    else:
        # Fallback text
        tb = slide4.shapes.add_textbox(Inches(7.3), Inches(2.0), Inches(5.0), Inches(4.0))
        tb.text_frame.text = "Raw Clinical Distributions (18 Labs & Vitals Generated from Latent Severity)"

    # Caption box
    cap_box = slide4.shapes.add_textbox(Inches(7.2), Inches(6.35), Inches(5.2), Inches(0.5))
    p_cap = cap_box.text_frame.paragraphs[0]
    p_cap.text = "Figure: Generated raw clinical variable distributions (Vincent SOFA inputs & vitals)."
    p_cap.font.size = Pt(9.5)
    p_cap.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 5: THE SOFA SCORE FORMULATION (VINCENT 1996 / SEPSIS-3)
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Clinical Grounding: The Vincent 1996 SOFA Score", "Section 2.1 | Physiological Modeling")

    add_card(slide5, Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.5))
    tb5 = slide5.shapes.add_textbox(Inches(1.1), Inches(1.55), Inches(11.133), Inches(5.1))
    tf5 = tb5.text_frame
    tf5.word_wrap = True

    p = tf5.paragraphs[0]
    p.text = "Sequential Organ Failure Assessment: SOFA_i = SUM(j=1 to 6) SOFA_ij in [0, 24]"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(10)

    sofa_organs = [
        ("1. Respiratory (PaO2/FiO2)", "Score 1: PF < 400 | Score 2: PF < 300\nScore 3: PF < 200 + Mech Vent | Score 4: PF < 100 + Mech Vent"),
        ("2. Coagulation (Platelets)", "Score 1: < 150 x10^3/uL | Score 2: < 100 x10^3/uL\nScore 3: < 50 x10^3/uL  | Score 4: < 20 x10^3/uL"),
        ("3. Liver (Bilirubin)", "Score 1: >= 1.2 mg/dL | Score 2: >= 2.0 mg/dL\nScore 3: >= 6.0 mg/dL | Score 4: >= 12.0 mg/dL"),
        ("4. Cardiovascular (MAP / Pressors)", "Score 1: MAP < 70 mmHg\nScore 2: Dopamine <= 5 mcg/kg/min\nScore 3: Dopamine > 5 or Norepi <= 0.1\nScore 4: Dopamine > 15 or Norepi > 0.1"),
        ("5. CNS (Glasgow Coma Scale)", "Score 1: GCS 13-14 | Score 2: GCS 10-12\nScore 3: GCS 6-9   | Score 4: GCS < 6"),
        ("6. Renal (Creatinine / Urine)", "Score 1: Creat >= 1.2 mg/dL\nScore 2: Creat >= 2.0 mg/dL\nScore 3: Creat >= 3.5 or Urine < 500 mL/d\nScore 4: Creat >= 5.0 or Urine < 200 mL/d")
    ]

    for idx, (org_t, org_d) in enumerate(sofa_organs):
        col = idx % 3
        row = idx // 3
        card_x = Inches(1.1 + col * 3.8)
        card_y = Inches(2.2 + row * 2.3)

        box_shape = slide5.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, card_x, card_y, Inches(3.6), Inches(2.1))
        box_shape.fill.solid()
        box_shape.fill.fore_color.rgb = CARD_BG_TINT
        box_shape.line.color.rgb = BORDER_COL

        tb_box = slide5.shapes.add_textbox(card_x + Inches(0.15), card_y + Inches(0.12), Inches(3.3), Inches(1.85))
        tf_box = tb_box.text_frame
        tf_box.word_wrap = True
        p_ot = tf_box.paragraphs[0]
        p_ot.text = org_t
        p_ot.font.size = Pt(12)
        p_ot.font.bold = True
        p_ot.font.color.rgb = DARK_TEAL
        p_ot.space_after = Pt(4)

        for line in org_d.split("\n"):
            p_od = tf_box.add_paragraph()
            p_od.text = line
            p_od.font.size = Pt(9.5)
            p_od.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 6: 48 DIURNAL ARRIVAL WINDOWS & TIME-OF-DAY DEMAND
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "48 Diurnal Windows & 24-Hour Time-of-Day Dynamics", "Section 2.1 | Episode Architecture")

    add_card(slide6, Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.5))
    tb6 = slide6.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.4), Inches(5.1))
    tf6 = tb6.text_frame
    tf6.word_wrap = True
    p = tf6.paragraphs[0]
    p.text = "Modeling Full-Day Hospital Rhythms"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(10)

    bullets6 = [
        ("48 Daily Half-Hour Slots", "Episode e maps to slot = e mod 48 (from 00:00-00:30 up to 23:30-24:00). Each window appears ~21 times across 1,000 episodes."),
        ("Sinusoidal Diurnal Demand Function", "d(h) = cos(2*pi*(h - 15) / 24) with midpoint hour h.\nPeak at 15:00 (3 PM, d=+1), Trough at 03:00 (3 AM, d=-1)."),
        ("Scenario Modulations", "Mean arrivals vary from 4.0 (02:30 AM) to 16.0 (02:30 PM). Afternoon congestion shifts probabilities toward high load (0-1 free beds)."),
        ("Window vs. Simulation Horizon", "The 30-min window defines the patient cohort. The simulation continues until ALL cohort patients are admitted (up to 72h max horizon)."),
        ("Length-Biased Census Carry-over", "Initial census starts at ~18/20 beds occupied. Hidden discharge times release beds asynchronously.")
    ]
    for b in bullets6:
        p1 = tf6.add_paragraph()
        p1.text = "• " + b[0]
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_DARK
        p2 = tf6.add_paragraph()
        p2.text = "   " + b[1]
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_after = Pt(6)

    # Right Image: windows_through_day.png
    add_card(slide6, Inches(6.9), Inches(1.4), Inches(5.633), Inches(5.5))
    fig_path = FIG_DIR / "windows_through_day.png"
    if fig_path.exists():
        slide6.shapes.add_picture(str(fig_path), Inches(7.05), Inches(1.6), width=Inches(5.333))

    cap_box = slide6.shapes.add_textbox(Inches(7.0), Inches(6.35), Inches(5.4), Inches(0.5))
    p_cap = cap_box.text_frame.paragraphs[0]
    p_cap.text = "Figure: Mean cohort arrivals and available beds across the 48 daily time slots."
    p_cap.font.size = Pt(9.5)
    p_cap.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 7: SEMI-MARKOV DECISION PROCESS (SMDP) FORMULATION
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "The Semi-Markov Decision Process (SMDP) Framework", "Section 2.2 | Formal Problem Formulation")

    add_card(slide7, Inches(0.8), Inches(1.4), Inches(5.6), Inches(5.5))
    tb7 = slide7.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.2), Inches(5.1))
    tf7 = tb7.text_frame
    tf7.word_wrap = True
    p = tf7.paragraphs[0]
    p.text = "Event-Driven Asynchronous Transitions"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(10)

    bullets7 = [
        ("Why Discrete Fixed Steps Fail", "Real medical events occur in continuous time. Decisions are only required when beds are free (B_t > 0) and the queue is non-empty (|Q_t| > 0)."),
        ("Priority Heap Scheduler (Chronological)", "Ties broken with strict clinical priority:\n1. Discharge (0) -> Bed becomes free\n2. Arrival (1) -> Enters waiting queue\n3. Deterioration (2) -> Severity jumps\n4. Window End (3) -> Cohort frozen"),
        ("Continuous-Time Discounting", "Discount factor: gamma^(dt / tau0), with gamma = 0.99 and tau0 = 1.0 h.\nBetween two distant events (dt = 3.5 h), gamma_eff = 0.99^3.5 = 0.965."),
        ("Multi-Patient Decision Epochs (dt = 0)", "When multiple beds are open, the agent admits sequentially at the exact same instant (dt = 0, gamma^0 = 1.0, zero discounting).")
    ]
    for b in bullets7:
        p1 = tf7.add_paragraph()
        p1.text = "• " + b[0]
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_DARK
        p2 = tf7.add_paragraph()
        p2.text = "   " + b[1]
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_after = Pt(6)

    # Right Image: mdp_diagram.png
    add_card(slide7, Inches(6.7), Inches(1.4), Inches(5.833), Inches(5.5))
    fig_path = FIG_DIR / "mdp_diagram.png"
    if fig_path.exists():
        slide7.shapes.add_picture(str(fig_path), Inches(6.85), Inches(1.7), width=Inches(5.5))

    cap_box = slide7.shapes.add_textbox(Inches(6.8), Inches(6.35), Inches(5.6), Inches(0.5))
    p_cap = cap_box.text_frame.paragraphs[0]
    p_cap.text = "Figure: Agent-Environment Interaction in the continuous event-driven SMDP."
    p_cap.font.size = Pt(9.5)
    p_cap.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 8: STATE SPACE REPRESENTATION
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "State Space Representation: The Active Patient Matrix", "Section 2.2 | State Engineering")

    add_card(slide8, Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.5))
    tb8 = slide8.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.4), Inches(5.1))
    tf8 = tb8.text_frame
    tf8.word_wrap = True
    p = tf8.paragraphs[0]
    p.text = "Active Matrix X_t in R^(N_t x 5)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(10)

    bullets8 = [
        ("Dynamic Row Count N_t = N_icu + N_q", "Only active patients (in ICU or waiting) enter the state. Discharged and unarrived patients are completely excluded. Patient IDs are never features."),
        ("Feature 1: Normalized SOFA", "SOFA / 24 in [0, 1]. Captures overall biological organ failure severity."),
        ("Feature 2: Expected Bed Occupancy", "Min(LOS_pred / 48.0 h, 5.0). Predicted total stay if waiting; predicted remaining stay if currently in ICU."),
        ("Feature 3: Normalized Waiting Time", "Min(W_i(t) / 8.0 h, 3.0). Exactly 0.0 for ICU occupants. Conveys queue urgency and fairness."),
        ("Feature 4: Predicted Deterioration Hazard", "Min(Hazard_pred / 0.3 h^-1, 3.0). Exactly 0.0 for ICU occupants."),
        ("Feature 5: Categorical Status (Embedding)", "1 = Waiting, 2 = ICU. Mapped to an 8-dim learned dense embedding."),
        ("Global Hospital Scalar: B_t / C", "Free bed fraction (20 - Occupied) / 20 fed into context aggregation.")
    ]
    for b in bullets8:
        p1 = tf8.add_paragraph()
        p1.text = "• " + b[0]
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_DARK
        p2 = tf8.add_paragraph()
        p2.text = "   " + b[1]
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_after = Pt(5)

    # Right Image: state_example.png
    add_card(slide8, Inches(6.9), Inches(1.4), Inches(5.633), Inches(5.5))
    fig_path = FIG_DIR / "state_example.png"
    if fig_path.exists():
        slide8.shapes.add_picture(str(fig_path), Inches(7.05), Inches(1.65), width=Inches(5.333))

    cap_box = slide8.shapes.add_textbox(Inches(7.0), Inches(6.35), Inches(5.4), Inches(0.5))
    p_cap = cap_box.text_frame.paragraphs[0]
    p_cap.text = "Figure: Snapshot of actual state tensor X_t fed into the neural policy network."
    p_cap.font.size = Pt(9.5)
    p_cap.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 9: ACTION SPACE & SEQUENTIAL MASKING MECHANISM
    # =========================================================================
    add_two_column_slide(
        title="Action Space & Sequential Masking Mechanism",
        cat="Section 2.2 | Action Space Design",
        left_title="Action Formulation: A_t = {ADMIT(i)} U {STOP}",
        left_bullets=[
            ("Why Not a Multi-Binary Vector (2^N)?", "Joint selection over N waiting patients yields 2^N combinations. With up to 29 candidates, 2^29 > 500 million actions—computationally infeasible and permutation-incompatible."),
            ("Sequential Masked Selection", "At each decision epoch, the agent selects one waiting patient row ADMIT(i), or chooses STOP. If beds remain free, it selects again at the same epoch (dt = 0)."),
            ("Action Set Definition", "A_t in {0, 1, ..., N_t - 1} U {STOP (action index N_t)}.\nEligible candidates correspond strictly to active waiting patients."),
            ("Hard Capacity Constraint Assertion", "Occupancy <= C (20 beds) is asserted at every transition. Violations = 0 across all 6,000 training episodes and all test evaluations.")
        ],
        right_title="Action Masking & Deferral Logic",
        right_bullets=[
            ("Action Mask Construction m_t", "Mask vector of size N_t + 1:\n- True for waiting patients if free beds B_t > 0.\n- False for ICU occupants and discharged patients.\n- True for STOP whenever an admission decision is offered."),
            ("Enforcing Infeasibility via -1e9 Masking", "Logits / Q-values of masked invalid actions are filled with -1e9 prior to softmax or argmax, guaranteeing zero illegal choices."),
            ("The Strategic Role of STOP", "Unlike greedy heuristics, RL agents can choose STOP to hold a bed open if high-risk, severe patients are predicted to arrive shortly."),
            ("Completion Guarantee", "Episodes terminate when all cohort patients are admitted (100% completion in test runs). 72h max horizon prevents deadlock.")
        ]
    )

    # =========================================================================
    # SLIDE 10: DUAL REWARD FORMULATION & EXACT INTEGRATION
    # =========================================================================
    add_two_column_slide(
        title="Dual Reward Formulations & Analytical Integration",
        cat="Section 2.3 | Reward Engineering",
        left_title="Reward 1: Aggregate Waiting Time (R1)",
        left_bullets=[
            ("Core Mathematical Definition", "R1(t) = - |Q_t| * dt - lambda_U * U_t\nCost equals the area under the queue curve over elapsed time dt."),
            ("Physical Meaning", "For example, 5 patients waiting for 2.0 hours produces 5 x 2.0 = 10 patient-hours of cost."),
            ("Cumulative Episode Alignment", "The sum of negative R1 over the entire episode exactly equals the total patient-hours spent waiting by the cohort."),
            ("Managerial Perspective", "Reward 1 is ideal for hospital managers seeking to maximize bed turnover and clear aggregate patient queues as fast as possible.")
        ],
        right_title="Reward 2: Severity-Weighted Waiting (R2)",
        right_bullets=[
            ("Core Mathematical Formulation", "R2(t) = - SUM(i in Q) INT_0^dt [ alpha * S_i + beta * min((W_i(t+s))/W_ref, 1) ] ds - lambda_U * U_t\nwith S_i = SOFA_i / 24, W_ref = 8.0 h."),
            ("Exact Analytical Closed-Form Solution", "Because SOFA and queue remain constant between events, the integral solves in closed-form without numerical step errors:\nINT = [ w_i*tau + 0.5*tau^2 ] / W_ref + (dt - tau)\nwhere tau = min(dt, max(W_ref - w_i, 0))."),
            ("Clinical Ethics & Prioritization", "Heavily penalizes delaying sickest patients (high SOFA) while preventing long-wait neglect through the beta term."),
            ("Tunable Clinical Dial", "(alpha, beta) coefficients allow adjusting hospital policy between pure throughput and strict clinical triage.")
        ]
    )

    # =========================================================================
    # SLIDE 11: NEURAL ARCHITECTURE: PERMUTATION-EQUIVARIANT SETNET
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Permutation-Equivariant SetNet Architecture", "Section 2.4 | Deep Sets Neural Network")

    add_card(slide11, Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.5))
    tb11 = slide11.shapes.add_textbox(Inches(1.1), Inches(1.55), Inches(11.133), Inches(5.1))
    tf11 = tb11.text_frame
    tf11.word_wrap = True

    p = tf11.paragraphs[0]
    p.text = "Deep Sets Architecture for Variable-Size Patient Queues"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(10)

    steps = [
        ("1. Input & Status Embedding", "Input: x_i in R^4 (SOFA, LOS, wait, hazard)\nCategorical status in {0, 1} passed through an nn.Embedding(2, 8). Concatenated input dimension = 12."),
        ("2. Shared Row Encoder MLP", "Shared 2-layer MLP(12 -> 64 -> 64) with ReLU activations. Each active patient row is mapped to a 64-dim latent embedding e_i."),
        ("3. Masked Pooling Operations", "Masked Mean-Pooling: captures average hospital patient profile.\nMasked Max-Pooling: captures worst-case patient severity.\nBoth are strictly permutation-invariant."),
        ("4. Context Aggregator", "Context MLP(64+64+1 -> 64) combines mean-pool, max-pool, and scalar free bed fraction B_t / C into global hospital context c."),
        ("5. Patient Q-Score Head", "MLP(64+64 -> 64 -> 1) concatenates [e_i, c] to produce Q(s, ADMIT(i)) for each patient row. Permutation-equivariant by construction."),
        ("6. STOP Head & Critic Head", "STOP MLP(64 -> 64 -> 1) maps context c to Q(s, STOP).\nValue Head MLP(64 -> 64 -> 1) maps context c to V(s) for PPO critic.")
    ]

    for idx, (st_t, st_d) in enumerate(steps):
        col = idx % 3
        row = idx // 3
        card_x = Inches(1.1 + col * 3.8)
        card_y = Inches(2.2 + row * 2.3)

        box_shape = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, card_x, card_y, Inches(3.6), Inches(2.1))
        box_shape.fill.solid()
        box_shape.fill.fore_color.rgb = CARD_BG_TINT
        box_shape.line.color.rgb = BORDER_COL

        tb_box = slide11.shapes.add_textbox(card_x + Inches(0.15), card_y + Inches(0.12), Inches(3.3), Inches(1.85))
        tf_box = tb_box.text_frame
        tf_box.word_wrap = True
        p_ot = tf_box.paragraphs[0]
        p_ot.text = st_t
        p_ot.font.size = Pt(12)
        p_ot.font.bold = True
        p_ot.font.color.rgb = DARK_TEAL
        p_ot.space_after = Pt(4)

        for line in st_d.split("\n"):
            p_od = tf_box.add_paragraph()
            p_od.text = line
            p_od.font.size = Pt(9.5)
            p_od.font.color.rgb = TEXT_DARK

    # =========================================================================
    # SLIDE 12: REINFORCEMENT LEARNING ALGORITHMS & TRAINING HARNESS
    # =========================================================================
    add_two_column_slide(
        title="Reinforcement Learning Algorithms Compared",
        cat="Section 2.5 | Algorithmic Design",
        left_title="Value-Based & Policy-Based Algorithms",
        left_bullets=[
            ("Deep Q-Network (DQN)", "Replay buffer (50k), target network update every 250 steps, Huber loss. Target: y = r + gamma^(dt / tau0) * max_a' Q_target(s', a'). Eps-greedy exploration with STOP-light probability 0.05."),
            ("Double DQN (DDQN)", "Decouples action selection from Q-value evaluation to stop overestimation: y = r + gamma^(dt / tau0) * Q_target(s', argmax_a' Q_online(s', a'))."),
            ("Proximal Policy Optimization (PPO)", "Masked categorical policy, duration-aware GAE (lambda=0.95), clipped surrogate objective (clip=0.2), value loss weight 0.5, entropy bonus 0.01. Updates every 32 episodes (4 epochs, batch 256)."),
            ("Transparent Heuristic Baselines", "1. Random (uniform over eligible waiting rows)\n2. Shortest-Predicted-First (SPF: greedy shortest stay)\n3. Highest-SOFA-First (strict medical triage)")
        ],
        right_title="Unified Training & Validation Protocol",
        right_bullets=[
            ("Identical Training Budget", "All learners trained for 6,000 episodes on the 700 training episodes across 5 seeds (0, 1, 2, 3, 4)."),
            ("Validation Checkpointing", "Evaluated every 200 episodes on 100 validation episodes. Checkpoint with highest validation return retained as best model (*.pt)."),
            ("Common Random Numbers (CRN)", "Arrival order-statistics and deterioration budgets pre-drawn to ensure identical episode realizations across policies."),
            ("Evaluation on Held-Out Test Set", "All final models evaluated on the SAME 200 held-out test episodes covering all 48 daily time windows.")
        ]
    )

    # =========================================================================
    # SLIDE 13: EXPERIMENT 1: BENCHMARK RESULTS (REWARD 1)
    # =========================================================================
    slide13 = prs.slides.add_slide(blank_layout)
    add_header(slide13, "Experiment 1: Benchmark Results on 200 Test Episodes", "Section 3.1 | Primary Test Benchmark")

    # Table on Left
    add_card(slide13, Inches(0.8), Inches(1.4), Inches(6.2), Inches(5.5))
    tb13 = slide13.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.8), Inches(5.1))
    tf13 = tb13.text_frame
    tf13.word_wrap = True

    p = tf13.paragraphs[0]
    p.text = "Held-Out Test Results (5 Seeds, Reward 1)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(8)

    # Add PowerPoint Table
    rows = [
        ["Policy", "Total Wait (p-h)", "Mean Wait", "High-SOFA", "Return"],
        ["Random", "60.40 +- 0.30", "4.90 h", "5.42 h", "-3.020"],
        ["DQN", "57.34 +- 0.71", "4.88 h", "6.76 h", "-2.867"],
        ["DDQN", "57.74 +- 0.63", "4.87 h", "6.43 h", "-2.887"],
        ["PPO", "59.37 +- 6.15", "4.82 h", "5.47 h", "-2.969"],
        ["SPF Heuristic", "54.98 (best)", "4.53 h", "6.43 h", "-2.749"],
        ["SOFA-First", "67.87 (worst)", "5.39 h", "2.98 h", "-3.393"]
    ]

    t_shape = slide13.shapes.add_table(7, 5, Inches(0.95), Inches(2.05), Inches(5.9), Inches(2.8))
    table = t_shape.table
    table.columns[0].width = Inches(1.5)
    table.columns[1].width = Inches(1.4)
    table.columns[2].width = Inches(1.0)
    table.columns[3].width = Inches(1.0)
    table.columns[4].width = Inches(1.0)

    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            p.font.size = Pt(9.5)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            else:
                p.font.color.rgb = TEXT_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 1 else CARD_BG_TINT

    # Summary bullet points under table
    tb_sub = slide13.shapes.add_textbox(Inches(0.95), Inches(4.95), Inches(5.9), Inches(1.8))
    tf_sub = tb_sub.text_frame
    tf_sub.word_wrap = True
    b_pts = [
        "All RL agents beat Random: DQN (-5.1%), DDQN (-4.4%), PPO (-1.7%).",
        "SPF Heuristic dominates pure delay (-9.0% vs Random) by clearing short stays.",
        "Highest-SOFA-first explodes total wait to 67.87 h (+12.4% penalty) because long severe stays block beds for everyone else.",
        "Zero capacity violations and 100% admission completion across all policies."
    ]
    for bp in b_pts:
        p = tf_sub.add_paragraph()
        p.text = "• " + bp
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_MUTED

    # Right Image: algorithm_comparison.png
    add_card(slide13, Inches(7.3), Inches(1.4), Inches(5.233), Inches(5.5))
    fig_path = FIG_DIR / "algorithm_comparison.png"
    if fig_path.exists():
        slide13.shapes.add_picture(str(fig_path), Inches(7.45), Inches(1.6), width=Inches(4.933))

    cap_box = slide13.shapes.add_textbox(Inches(7.4), Inches(6.35), Inches(5.0), Inches(0.5))
    p_cap = cap_box.text_frame.paragraphs[0]
    p_cap.text = "Figure: Algorithm comparison on held-out test episodes (multi-metric error bars)."
    p_cap.font.size = Pt(9.5)
    p_cap.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 14: EXPERIMENT 2: SELECTION CRITERIA (WHY DDQN WON)
    # =========================================================================
    slide14 = prs.slides.add_slide(blank_layout)
    add_header(slide14, "Experiment 2: Algorithm Selection (Why DDQN Won)", "Section 3.2 | Model Selection Analysis")

    add_card(slide14, Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.5))
    tb14 = slide14.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.4), Inches(5.1))
    tf14 = tb14.text_frame
    tf14.word_wrap = True
    p = tf14.paragraphs[0]
    p.text = "Multi-Criteria Evaluation Matrix"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(8)

    crit_rows = [
        ["Criterion", "DQN", "DDQN (Selected)", "PPO"],
        ["Seed Robustness (SD)", "0.71 p-h", "0.63 p-h (Lowest)", "6.15 p-h (Unstable)"],
        ["Seeds Better than Random", "5 / 5 seeds", "5 / 5 seeds (100%)", "3 / 5 seeds (60%)"],
        ["Convergence Episode", "1,667 ep", "500 ep (Fastest)", "2,133 ep"],
        ["Training Wall-Clock Time", "1,338 +- 53 s", "1,371 +- 87 s", "312 +- 56 s (Fastest)"],
        ["Test Total Wait Difference", "+0.39 p-h", "Reference", "+2.03 p-h"]
    ]
    t_shape = slide14.shapes.add_table(6, 4, Inches(0.95), Inches(2.05), Inches(5.5), Inches(2.5))
    table = t_shape.table
    table.columns[0].width = Inches(1.8)
    table.columns[1].width = Inches(1.1)
    table.columns[2].width = Inches(1.4)
    table.columns[3].width = Inches(1.2)

    for r_idx, row in enumerate(crit_rows):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            p.font.size = Pt(9.5)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            else:
                p.font.color.rgb = TEXT_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 1 else CARD_BG_TINT

    # Explanation bullets below
    tb_crit = slide14.shapes.add_textbox(Inches(0.95), Inches(4.65), Inches(5.5), Inches(2.1))
    tf_c = tb_crit.text_frame
    tf_c.word_wrap = True
    c_pts = [
        "DDQN Winner on Robustness: Lowest seed SD (0.63) and 100% of seeds beat Random.",
        "DQN Tied on Performance: DQN vs DDQN difference is 0.39 p-h (95% CI [-0.02, 0.82]), statistically not significant.",
        "PPO Ruled Out for Instability: Despite fast GPU/CPU training, seeds 2 and 4 collapsed into poor local deterministic orderings (total wait > 63h)."
    ]
    for cp in c_pts:
        p = tf_c.add_paragraph()
        p.text = "• " + cp
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_MUTED

    # Right Image: convergence_all.png
    add_card(slide14, Inches(6.9), Inches(1.4), Inches(5.633), Inches(5.5))
    fig_path = FIG_DIR / "convergence_all.png"
    if fig_path.exists():
        slide14.shapes.add_picture(str(fig_path), Inches(7.05), Inches(1.6), width=Inches(5.333))

    cap_box = slide14.shapes.add_textbox(Inches(7.0), Inches(6.35), Inches(5.4), Inches(0.5))
    p_cap = cap_box.text_frame.paragraphs[0]
    p_cap.text = "Figure: Validation learning curves across 6,000 episodes for DQN, DDQN, and PPO."
    p_cap.font.size = Pt(9.5)
    p_cap.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 15: CASE STUDY: TEST EPISODE 21 IN ACTION
    # =========================================================================
    slide15 = prs.slides.add_slide(blank_layout)
    add_header(slide15, "Case Study: Allocation Dynamics in Test Episode 21", "Section 3.1 | Interpretability & Policy Rollout")

    add_card(slide15, Inches(0.8), Inches(1.4), Inches(5.4), Inches(5.5))
    tb15 = slide15.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.0), Inches(5.1))
    tf15 = tb15.text_frame
    tf15.word_wrap = True
    p = tf15.paragraphs[0]
    p.text = "Episode 21 Profile & Decision Trace"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(10)

    bullets15 = [
        ("Episode Profile", "Slot 21 (10:30-11:00 AM window), medium congestion scenario, 3 free beds at start, 9 cohort arrivals."),
        ("Selection Rule", "Pre-specified rule: first medium-congestion test episode with 6-9 arrivals and >= 2 free beds (unbiased selection)."),
        ("Learned Admission Dynamics", "DDQN admits patients with short predicted stays early to free up beds rapidly for incoming patients."),
        ("Prioritizing Turnover", "Long-stay patients with high SOFA wait longer until bed release events create slack capacity."),
        ("Episode-Level Variance Note", "In this specific episode, DDQN total wait (41.9 p-h) was slightly higher than Random (41.5 p-h), highlighting that RL advantages manifest as aggregate statistical expectations across stochastic episodes.")
    ]
    for b in bullets15:
        p1 = tf15.add_paragraph()
        p1.text = "• " + b[0]
        p1.font.size = Pt(12)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_DARK
        p2 = tf15.add_paragraph()
        p2.text = "   " + b[1]
        p2.font.size = Pt(10.5)
        p2.font.color.rgb = TEXT_MUTED
        p2.space_after = Pt(6)

    # Right Image: agent_snapshots.png
    add_card(slide15, Inches(6.5), Inches(1.4), Inches(6.033), Inches(5.5))
    fig_path = FIG_DIR / "agent_snapshots.png"
    if fig_path.exists():
        slide15.shapes.add_picture(str(fig_path), Inches(6.65), Inches(1.55), width=Inches(5.733))

    cap_box = slide15.shapes.add_textbox(Inches(6.6), Inches(6.35), Inches(5.8), Inches(0.5))
    p_cap = cap_box.text_frame.paragraphs[0]
    p_cap.text = "Figure: Episode 21 occupancy over time, queue decay, and patient admission Gantt charts."
    p_cap.font.size = Pt(9.5)
    p_cap.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 16: EXPERIMENT 3: REWARD 1 VS REWARD 2 (CLINICAL PRIORITIZATION)
    # =========================================================================
    slide16 = prs.slides.add_slide(blank_layout)
    add_header(slide16, "Experiment 3: Reward 1 vs. Reward 2 (Clinical Prioritization)", "Section 3.3 | Ethical & Severity Trade-off")

    add_card(slide16, Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.5))
    tb16 = slide16.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(5.4), Inches(5.1))
    tf16 = tb16.text_frame
    tf16.word_wrap = True
    p = tf16.paragraphs[0]
    p.text = "The Throughput vs. Clinical Acuity Trade-off"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(8)

    r_table = [
        ["Metric", "DDQN (Reward 1)", "DDQN (Reward 2)", "Paired Difference"],
        ["High-SOFA Wait", "6.43 +- 0.42 h", "5.30 +- 0.65 h", "-1.13 h (-17.6%)*"],
        ["Total Waiting", "57.74 +- 0.63 p-h", "58.87 +- 1.40 p-h", "+1.14 p-h (+2.0%)"],
        ["Max Wait Time", "8.61 +- 0.24 h", "8.75 +- 0.37 h", "+0.14 h (n.s.)"],
        ["Severity Cost", "69.84 +- 0.40", "70.23 +- 0.84", "+0.39 (n.s.)"],
        ["Return under R1", "-2.89", "-2.94", "R1 policy wins"],
        ["Return under R2", "-5.45", "-5.43", "R2 policy wins"]
    ]
    t_shape = slide16.shapes.add_table(7, 4, Inches(0.95), Inches(2.05), Inches(5.5), Inches(2.6))
    table = t_shape.table
    table.columns[0].width = Inches(1.5)
    table.columns[1].width = Inches(1.3)
    table.columns[2].width = Inches(1.3)
    table.columns[3].width = Inches(1.4)

    for r_idx, row in enumerate(r_table):
        for c_idx, val in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c_idx > 0 else PP_ALIGN.LEFT
            p.font.size = Pt(9.5)
            if r_idx == 0:
                p.font.bold = True
                p.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            else:
                p.font.color.rgb = TEXT_DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if r_idx % 2 == 1 else CARD_BG_TINT

    # Trade-off callout
    tb_call = slide16.shapes.add_textbox(Inches(0.95), Inches(4.8), Inches(5.5), Inches(1.9))
    tf_call = tb_call.text_frame
    tf_call.word_wrap = True
    c_pts2 = [
        "Substantial Clinical Benefit: Reward 2 reduces high-SOFA waiting by 17.6% (95% CI [-1.31, -0.95]), preventing organ deterioration.",
        "Minimal Operational Cost: Overall delay increases by only 2.0% (+1.14 patient-hours).",
        "Clinical Slider: Tuning alpha/beta acts as an explicit dial between queue throughput and triage ethics."
    ]
    for cp in c_pts2:
        p = tf_call.add_paragraph()
        p.text = "• " + cp
        p.font.size = Pt(10)
        p.font.color.rgb = TEXT_MUTED

    # Right Images: reward_comparison.png and reward_sensitivity.png
    add_card(slide16, Inches(6.9), Inches(1.4), Inches(5.633), Inches(5.5))
    fig_path1 = FIG_DIR / "reward_comparison.png"
    fig_path2 = FIG_DIR / "reward_sensitivity.png"

    if fig_path1.exists() and fig_path2.exists():
        slide16.shapes.add_picture(str(fig_path1), Inches(7.05), Inches(1.5), width=Inches(5.333))
        slide16.shapes.add_picture(str(fig_path2), Inches(7.45), Inches(3.9), width=Inches(4.5))

    cap_box = slide16.shapes.add_textbox(Inches(7.0), Inches(6.35), Inches(5.4), Inches(0.5))
    p_cap = cap_box.text_frame.paragraphs[0]
    p_cap.text = "Figure: Reward 1 vs 2 performance (top) and (alpha, beta) sensitivity Pareto curve (bottom)."
    p_cap.font.size = Pt(9.0)
    p_cap.font.color.rgb = TEXT_MUTED

    # =========================================================================
    # SLIDE 17: TRANSPARENT DISCLOSURES & KNOWN LIMITATIONS
    # =========================================================================
    add_two_column_slide(
        title="Documented Assumptions & Academic Disclosures",
        cat="Section 4 | Limitations & Integrity",
        left_title="Known Limitations & Simulator Flaws",
        left_bullets=[
            ("Synthetic & Uncalibrated Data", "Data is synthetic, not MIMIC-IV. Findings represent mathematical proof-of-concept simulation, NOT clinical evidence."),
            ("Stay Predictor Overestimation Bias", "Total-stay GBR predictor was trained on arrivals and census patients together. For waiting arrivals, predicted stays are inflated by ~3.2x.\nHowever, it strictly preserves rank (Spearman 0.88), so policy input ranking remains consistent."),
            ("Window Independence", "Each of the 48 daily slots is drawn independently; census does not carry over between consecutive windows of the same day."),
            ("No Pre-emption / Static Bed Capacity", "Admitted patients cannot be bumped out of ICU beds; staffing, equipment, and ward transfer constraints are unmodelled.")
        ],
        right_title="Academic Integrity & Future Directions",
        right_bullets=[
            ("Iterative Development Disclosure", "Run 1 (10:00 AM window only) failed to beat Random. Scenario parameters and training budgets were revised transparently on validation sets to make the problem learnable."),
            ("Strict Leakage Prevention Validated", "Observable vs hidden files strictly quarantined; 21/21 automated leakage checks passed; predictors fitted on train-only out-of-fold folds."),
            ("Drop-in Replacement for MIMIC-IV", "The file schema (snapshots, arrival queues, census files) was designed explicitly for seamless swap-in of real hospital databases."),
            ("Future Roadmap", "1. Chained continuous daily windows\n2. Multi-hospital transfer networks\n3. Off-policy evaluation (OPE) with clinical advisory review.")
        ]
    )

    # =========================================================================
    # SLIDE 18: CONCLUSION & DEFENSE Q&A / TAKEAWAYS
    # =========================================================================
    slide18 = prs.slides.add_slide(blank_layout)
    add_header(slide18, "Conclusions & Examination Defense Preparation", "Section 5 | Summary & Examination Q&A")

    add_card(slide18, Inches(0.8), Inches(1.4), Inches(11.733), Inches(5.5))
    tb18 = slide18.shapes.add_textbox(Inches(1.1), Inches(1.55), Inches(11.133), Inches(5.1))
    tf18 = tb18.text_frame
    tf18.word_wrap = True

    p = tf18.paragraphs[0]
    p.text = "Key Takeaways & Prepared Answers for Professor Questions"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = NAVY
    p.space_after = Pt(10)

    qa_list = [
        ("Takeaway 1: SMDP Formulation", "Event-driven semi-Markov modeling matches realistic asynchronous hospital dynamics far better than artificial discrete time-steps."),
        ("Takeaway 2: Deep Sets Viability", "Permutation-equivariant SetNet solves variable queue sizing without ordering bias or combinatorial action explosion."),
        ("Takeaway 3: Robustness Over Speed", "DDQN is the superior choice for clinical environments due to high seed stability (SD=0.63) and overestimation resistance."),
        ("Q: Why did SPF heuristic beat RL on R1?", "Queueing theory proves Shortest-Job-First minimizes mean wait in single-stage queues. However, SPF starves severe patients; RL is essential once clinical equity (Reward 2) is introduced."),
        ("Q: Why RL instead of Linear Programming?", "Integer Linear Programming requires knowing all future arrival times and stay lengths in advance. RL operates under real-time stochastic uncertainty."),
        ("Q: What is the main clinical impact?", "Reward 2 achieves an 18% reduction in waiting time for critically ill patients (SOFA >= 11) at the cost of only 2% additional aggregate delay.")
    ]

    for idx, (q_t, q_d) in enumerate(qa_list):
        col = idx % 3
        row = idx // 3
        card_x = Inches(1.1 + col * 3.8)
        card_y = Inches(2.2 + row * 2.3)

        box_shape = slide18.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, card_x, card_y, Inches(3.6), Inches(2.1))
        box_shape.fill.solid()
        box_shape.fill.fore_color.rgb = CARD_BG_TINT
        box_shape.line.color.rgb = BORDER_COL

        tb_box = slide18.shapes.add_textbox(card_x + Inches(0.15), card_y + Inches(0.12), Inches(3.3), Inches(1.85))
        tf_box = tb_box.text_frame
        tf_box.word_wrap = True
        p_ot = tf_box.paragraphs[0]
        p_ot.text = q_t
        p_ot.font.size = Pt(11.5)
        p_ot.font.bold = True
        p_ot.font.color.rgb = DARK_TEAL
        p_ot.space_after = Pt(4)

        for line in q_d.split("\n"):
            p_od = tf_box.add_paragraph()
            p_od.text = line
            p_od.font.size = Pt(9.5)
            p_od.font.color.rgb = TEXT_DARK

    # Save presentation
    prs.save(str(OUTPUT_PATH))
    prs.save(str(ROOT_OUTPUT_PATH))
    print(f"Presentation successfully created and saved to:\n  {OUTPUT_PATH}\n  {ROOT_OUTPUT_PATH}")

if __name__ == "__main__":
    create_presentation()
