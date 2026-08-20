const API_URL = "http://127.0.0.1:8000";


/* =========================
   GENERIC API REQUEST
========================= */

async function apiRequest(endpoint, options = {}) {

    try {

        const response = await fetch(API_URL + endpoint, {
            headers: {
                "Content-Type": "application/json",
                ...(options.headers || {})
            },
            ...options
        });

        if (!response.ok) {

            let errorMessage = `API error: ${response.status}`;

            try {
                const errorData = await response.json();

                if (errorData.detail) {
                    errorMessage = errorData.detail;
                }

            } catch (_) {
                // Response wasn't JSON
            }

            throw new Error(errorMessage);
        }

        return await response.json();

    } catch (error) {

        console.error("Backend request failed:", error);

        throw error;
    }
}


/* =========================
   HEALTH
========================= */

async function checkBackendHealth() {

    return await apiRequest("/health");
}


/* =========================
   STUDENT
========================= */

async function createStudent(displayName) {

    return await apiRequest("/students", {
        method: "POST",

        body: JSON.stringify({
            display_name: displayName
        })
    });
}


/* =========================
   LEARNER PROFILE
========================= */

async function getProfile(studentId, topic) {

    return await apiRequest(
        `/profile/${encodeURIComponent(studentId)}/${encodeURIComponent(topic)}`
    );
}


/* =========================
   DIAGNOSTIC
========================= */

async function startDiagnostic(topic) {

    return await apiRequest(
        `/diagnostic/${encodeURIComponent(topic)}`
    );
}


async function submitDiagnostic(
    studentId,
    topic,
    responses
) {

    return await apiRequest("/diagnostic/submit", {

        method: "POST",

        body: JSON.stringify({

            student_id: studentId,
            topic: topic,
            responses: responses

        })
    });
}


/* =========================
   AI TUTOR
========================= */

async function teach(
    studentId,
    topic,
    struggling = false,
    priorExplanationSummary = null
) {

    return await apiRequest("/tutor/teach", {

        method: "POST",

        body: JSON.stringify({

            student_id: studentId,
            topic: topic,
            struggling: struggling,
            prior_explanation_summary:
                priorExplanationSummary

        })
    });
}


/* =========================
   ASSESSMENT
========================= */

async function getAssessmentQuestion(
    studentId,
    topic
) {

    return await apiRequest("/assessment/question", {

        method: "POST",

        body: JSON.stringify({

            student_id: studentId,
            topic: topic

        })
    });
}


async function submitAssessmentAnswer(
    studentId,
    topic,
    question,
    correctAnswer,
    studentAnswer
) {

    return await apiRequest("/assessment/answer", {

        method: "POST",

        body: JSON.stringify({

            student_id: studentId,
            topic: topic,
            question: question,
            correct_answer: correctAnswer,
            student_answer: studentAnswer

        })
    });
}


/* =========================
   NEXT ACTION
========================= */

async function getNextAction(
    studentId,
    topic
) {

    return await apiRequest(
        `/next-action/${encodeURIComponent(studentId)}/${encodeURIComponent(topic)}`
    );
}