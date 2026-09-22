"""
generate_resumes.py
--------------------
Creates ~50 SYNTHETIC resumes covering a mix of roles (PM, software engineer,
customer support, sales, designer, data analyst). Nothing here is a real person.

PM-relevant "why": the whole point of this demo is to prove that SEMANTIC
search beats plain keyword search. To prove that, the data has to contain
candidates who describe the same underlying skill in very different words
(e.g. "led checkout redesign" vs "owned payments product area" vs "drove
end-to-end ownership of the online purchase flow"). If every resume used the
same vocabulary, TF-IDF/BM25 keyword matching would look just as good as
embeddings, and the demo wouldn't teach anything.

Each resume is stored as a dict with:
  - candidate_id: stable synthetic id
  - full_name: fictional name
  - raw_text: full resume text with Experience / Skills / Education sections
  - date_received: ISO date, randomly spread over the last ~18 months
  - true_role_family: ground truth label used later ONLY for evaluation/labeling,
    never fed into the search system itself (that would be cheating).
"""

import json
import random
from datetime import datetime, timedelta

random.seed(42)  # reproducible synthetic dataset

OUT_PATH = "data/resumes.json"

FIRST_NAMES = [
    "Maria", "James", "Aisha", "Wei", "Carlos", "Priya", "Tom", "Fatima",
    "Liam", "Sofia", "Noah", "Yuki", "Omar", "Grace", "Diego", "Hana",
    "Ethan", "Zara", "Lucas", "Mei", "Amara", "Jack", "Layla", "Sam",
    "Nina", "David", "Chidi", "Elena", "Ravi", "Claire",
]
LAST_NAMES = [
    "Nguyen", "Smith", "Patel", "Garcia", "Kim", "Johnson", "Okafor",
    "Rossi", "Muller", "Tanaka", "Lopez", "Anderson", "Singh", "Dubois",
    "Kowalski", "Silva", "Haddad", "Larsen", "Brown", "Chen",
]

COMPANIES = [
    "Brightloop", "Nimbus Retail", "Fernway Labs", "Cobalt Health",
    "Pebblestone Financial", "Aurora Systems", "GreenPath Logistics",
    "Meadowbrook Software", "Northstar Analytics", "Vantage Commerce",
    "Clearwave Media", "Ironbridge Robotics", "Sunlit Foods",
    "Basecamp Mobility", "Quietwork Tools", "Harborlight Insurance",
]

# --- Role templates -----------------------------------------------------
# Each template gives several PHRASING VARIANTS per bullet so that the same
# underlying skill is described differently across candidates. This is the
# key ingredient for testing semantic vs keyword search.

PM_BULLETS = [
    [
        "Led the checkout redesign that reduced cart abandonment by 18%",
        "Owned the payments product area end-to-end, cutting drop-off at purchase by nearly a fifth",
        "Drove a full overhaul of the online purchase flow, improving completion rates across mobile and web",
    ],
    [
        "Ran discovery interviews with 40+ customers to prioritize the onboarding roadmap",
        "Partnered with UX research to talk to dozens of new users and reshape the first-week experience",
        "Synthesized customer feedback loops to guide the new-user activation roadmap",
    ],
    [
        "Shipped a self-serve analytics dashboard used by 200+ internal stakeholders",
        "Launched an internal reporting tool adopted company-wide by ops and finance teams",
        "Delivered a metrics workspace that replaced manual spreadsheet reporting for the whole org",
    ],
    [
        "Defined and tracked north-star metrics for the growth team, presenting weekly to leadership",
        "Owned the growth team's KPI framework and reported results to the executive staff",
        "Built the measurement framework growth leadership used to judge experiment success",
    ],
    [
        "Wrote PRDs and managed a cross-functional team of 6 engineers and 2 designers",
        "Coordinated a squad of engineers and designers from spec to launch",
        "Acted as the single point of accountability for a multi-discipline product squad",
    ],
    [
        "Ran A/B tests on pricing tiers that increased conversion by 9%",
        "Experimented with subscription pricing structures, lifting sign-up conversion meaningfully",
        "Used controlled experiments to validate a new pricing model before company-wide rollout",
    ],
]
PM_SKILLS = [
    "Roadmapping, Prioritization (RICE), SQL, Figma, A/B Testing, Stakeholder Management",
    "Product Strategy, User Research, JIRA, Amplitude, Cross-functional Leadership",
    "Agile/Scrum, Data Analysis, Competitive Analysis, PRD Writing, Pricing Strategy",
    "OKRs, Customer Discovery, Mixpanel, SQL, Go-to-Market Planning",
]

ENG_BULLETS = [
    [
        "Rebuilt the checkout service in Go, cutting p95 latency by 40%",
        "Rewrote the payments backend for performance, dropping tail latency significantly",
        "Migrated the purchase-flow API to a faster service architecture",
    ],
    [
        "Designed and shipped a microservices migration from a legacy monolith",
        "Led the decomposition of a monolithic codebase into independently deployable services",
        "Architected the transition off the old monolith onto microservices",
    ],
    [
        "Built CI/CD pipelines that cut deploy time from 45 minutes to 6",
        "Automated the release process, shrinking deployment time by nearly 90%",
        "Set up continuous delivery tooling that made releases fast and routine",
    ],
    [
        "Implemented a recommendation engine using collaborative filtering in Python",
        "Built a Python-based recommender system to personalize the product feed",
        "Developed ML-driven content ranking for the homepage feed",
    ],
    [
        "Mentored 3 junior engineers and led weekly code reviews",
        "Grew junior team members through regular code review and pairing",
        "Took ownership of onboarding and mentoring new engineering hires",
    ],
]
ENG_SKILLS = [
    "Python, Go, PostgreSQL, Kubernetes, AWS, REST APIs",
    "JavaScript, React, Node.js, Docker, CI/CD, GraphQL",
    "Java, Spring Boot, Microservices, Kafka, System Design",
    "Python, Machine Learning, Pandas, scikit-learn, SQL",
]

SUPPORT_BULLETS = [
    [
        "Resolved 60+ customer tickets daily with a 96% satisfaction score",
        "Handled a high volume of daily support requests while keeping CSAT above 95%",
        "Maintained top-tier satisfaction ratings across a heavy daily ticket load",
    ],
    [
        "Wrote and maintained the help center knowledge base, reducing repeat tickets by 20%",
        "Authored self-serve documentation that cut down repetitive support volume",
        "Built out the customer-facing FAQ library to deflect common questions",
    ],
    [
        "Trained 8 new support hires on the ticketing system and escalation process",
        "Onboarded new customer support staff onto tools and escalation paths",
        "Led new-hire training sessions for the support team's tooling and workflow",
    ],
    [
        "Identified a recurring billing bug and escalated it to engineering, preventing 500+ future tickets",
        "Flagged a systemic billing issue to engineering that headed off a wave of complaints",
        "Caught a pattern in payment-related complaints and routed it to the right engineering team",
    ],
]
SUPPORT_SKILLS = [
    "Zendesk, Intercom, Customer Empathy, Conflict Resolution, SLA Management",
    "Salesforce Service Cloud, Live Chat, Escalation Handling, Knowledge Base Writing",
    "Freshdesk, Ticket Triage, Onboarding, Process Documentation",
]

SALES_BULLETS = [
    [
        "Closed $1.2M in new annual recurring revenue across 30 enterprise accounts",
        "Generated over a million dollars in new ARR from enterprise deals",
        "Drove significant new recurring revenue growth through enterprise account wins",
    ],
    [
        "Built and ran the outbound prospecting playbook for the SMB segment",
        "Owned top-of-funnel outreach strategy for small business customers",
        "Designed the cold outreach process that fed the SMB pipeline",
    ],
    [
        "Negotiated and closed a multi-year contract with a Fortune 500 logistics company",
        "Secured a long-term agreement with a major enterprise logistics customer",
        "Landed a large multi-year deal with a top-tier logistics client",
    ],
    [
        "Consistently exceeded quota by 120%+ for six consecutive quarters",
        "Beat sales targets by more than 20% for a year and a half straight",
        "Delivered quota over-attainment across six straight quarters",
    ],
]
SALES_SKILLS = [
    "Salesforce, Outbound Prospecting, Negotiation, Account Management, Forecasting",
    "HubSpot, Enterprise Sales, Contract Negotiation, Pipeline Management",
    "Cold Calling, SMB Sales, CRM Hygiene, Upselling",
]

DESIGN_BULLETS = [
    [
        "Redesigned the checkout UI, improving task completion in usability tests by 25%",
        "Overhauled the purchase-flow interface, raising usability test success rates",
        "Simplified the buy-flow screens after multiple rounds of user testing",
    ],
    [
        "Built and maintained the company's design system component library in Figma",
        "Owned the shared UI component library used across product teams",
        "Standardized reusable design components adopted org-wide",
    ],
    [
        "Ran moderated usability studies with 15 participants per release cycle",
        "Conducted regular in-depth user testing sessions ahead of each release",
        "Led moderated research rounds to validate designs before shipping",
    ],
]
DESIGN_SKILLS = [
    "Figma, Design Systems, Usability Testing, Prototyping, Accessibility (WCAG)",
    "Sketch, User Research, Interaction Design, Wireframing",
]

DATA_BULLETS = [
    [
        "Built a churn prediction model that improved retention team targeting by 15%",
        "Developed a predictive model to flag at-risk customers for the retention team",
        "Created a machine learning model that improved how retention efforts were targeted",
    ],
    [
        "Automated weekly executive reporting, saving the team ~10 hours per week",
        "Replaced manual leadership reports with an automated pipeline",
        "Cut hours of manual report-building by automating the weekly exec dashboard",
    ],
    [
        "Partnered with product to define and instrument key funnel metrics",
        "Worked closely with the PM team to define event tracking for the funnel",
        "Set up analytics instrumentation for core product funnels",
    ],
]
DATA_SKILLS = [
    "SQL, Python, Tableau, A/B Testing, Statistics",
    "R, dbt, Looker, Experimentation, Data Modeling",
]

ROLE_LIBRARY = {
    "Product Manager": {
        "titles": ["Product Manager", "Senior Product Manager", "Associate Product Manager"],
        "bullets": PM_BULLETS,
        "skills": PM_SKILLS,
    },
    "Software Engineer": {
        "titles": ["Software Engineer", "Senior Software Engineer", "Backend Engineer"],
        "bullets": ENG_BULLETS,
        "skills": ENG_SKILLS,
    },
    "Customer Support": {
        "titles": ["Customer Support Specialist", "Support Team Lead", "Customer Success Associate"],
        "bullets": SUPPORT_BULLETS,
        "skills": SUPPORT_SKILLS,
    },
    "Sales": {
        "titles": ["Account Executive", "Sales Development Representative", "Enterprise Sales Manager"],
        "bullets": SALES_BULLETS,
        "skills": SALES_SKILLS,
    },
    "Product Designer": {
        "titles": ["Product Designer", "UX Designer", "Senior Product Designer"],
        "bullets": DESIGN_BULLETS,
        "skills": DESIGN_SKILLS,
    },
    "Data Analyst": {
        "titles": ["Data Analyst", "Business Intelligence Analyst", "Analytics Associate"],
        "bullets": DATA_BULLETS,
        "skills": DATA_SKILLS,
    },
}

SCHOOLS = [
    "State University", "Lakeside College", "Northfield Institute of Technology",
    "Riverbend University", "Maple City College", "Union Polytechnic",
]
DEGREES = [
    "B.S. in Business Administration", "B.A. in Communications",
    "B.S. in Computer Science", "B.S. in Economics", "B.A. in Psychology",
    "M.S. in Information Systems",
]


def random_date_within_18_months():
    today = datetime(2026, 9, 22)
    days_back = random.randint(0, 548)  # ~18 months
    return (today - timedelta(days=days_back)).strftime("%Y-%m-%d")


def build_experience_section(role_family, num_jobs):
    lines = []
    lib = ROLE_LIBRARY[role_family]
    used_companies = random.sample(COMPANIES, num_jobs)
    for i, company in enumerate(used_companies):
        title = random.choice(lib["titles"])
        years_ago_start = 1 + i * 2 + random.randint(0, 1)
        years_ago_end = years_ago_start - 2 if i > 0 else 0
        date_str = f"{2026 - years_ago_start} - {'Present' if years_ago_end == 0 else 2026 - years_ago_end}"
        lines.append(f"{title}, {company} ({date_str})")
        # pick 2-3 distinct bullet groups, one phrasing variant from each
        groups = random.sample(lib["bullets"], min(3, len(lib["bullets"])))
        for group in groups:
            lines.append(f"- {random.choice(group)}")
        lines.append("")
    return "\n".join(lines).strip()


def build_resume(candidate_id, role_family):
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    full_name = f"{first} {last}"
    lib = ROLE_LIBRARY[role_family]

    num_jobs = random.choice([1, 2, 2, 3])
    experience = build_experience_section(role_family, num_jobs)
    skills = random.choice(lib["skills"])
    school = random.choice(SCHOOLS)
    degree = random.choice(DEGREES)
    grad_year = random.randint(2008, 2023)

    raw_text = f"""{full_name}
{role_family} candidate

Experience
{experience}

Skills
{skills}

Education
{degree}, {school}, {grad_year}
"""
    return {
        "candidate_id": candidate_id,
        "full_name": full_name,
        "raw_text": raw_text.strip(),
        "date_received": random_date_within_18_months(),
        "true_role_family": role_family,  # ground truth, kept OUT of search index
    }


def main():
    role_families = list(ROLE_LIBRARY.keys())
    resumes = []
    n = 52
    cid = 1
    # Roughly even spread across role families, with PM slightly oversampled
    # since this is a PM-focused portfolio demo.
    weights = {"Product Manager": 14, "Software Engineer": 10, "Customer Support": 8,
               "Sales": 8, "Product Designer": 6, "Data Analyst": 6}
    for role, count in weights.items():
        for _ in range(count):
            resumes.append(build_resume(f"C{cid:03d}", role))
            cid += 1

    random.shuffle(resumes)
    import os
    os.makedirs("data", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(resumes, f, indent=2)

    print(f"Generated {len(resumes)} synthetic resumes -> {OUT_PATH}")
    from collections import Counter
    print("Role mix:", Counter(r["true_role_family"] for r in resumes))


if __name__ == "__main__":
    main()
