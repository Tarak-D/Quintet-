/* =========================================================
   QUINTET ASSESSMENT
========================================================= */

let currentAssessment = null;


/* =========================================================
   DOM ELEMENTS
========================================================= */

const assessmentContainer =
    document.getElementById("assessment");


/* =========================================================
   START ASSESSMENT
========================================================= */

async function startAssessment() {

    if (!currentStudent) {
        alert("Student profile is not ready yet.");
        return;
    }

    const container =
        document.getElementById("assessment");

    if (!container) {
        return;
    }

    container.innerHTML = `
        <div class="card">
            <h2>Assessment</h2>
            <p>Generating your question...</p>
        </div>
    `;

    try {

        const question =
            await getAssessmentQuestion(
                currentStudent.student_id,
                currentTopic
            );

        currentAssessment = question;

        renderAssessmentQuestion(question);

    } catch (error) {

        console.error(
            "Assessment generation failed:",
            error
        );

        container.innerHTML = `
            <div class="card">
                <h2>Assessment</h2>
                <p>
                    Unable to generate a question.
                    Please try again.
                </p>

                <button onclick="startAssessment()">
                    Try Again
                </button>
            </div>
        `;
    }
}


/* =========================================================
   RENDER QUESTION
========================================================= */

function renderAssessmentQuestion(question) {

    const container =
        document.getElementById("assessment");

    if (!container) {
        return;
    }

    const questionText =
        question.question || "Question unavailable.";

    container.innerHTML = `
        <div class="card assessment-card">

            <div class="assessment-header">

                <div>
                    <h2>Assessment</h2>

                    <p>
                        Topic:
                        <strong>
                            ${escapeHTML(currentTopic)}
                        </strong>
                    </p>
                </div>

            </div>


            <div class="question-box">

                <h3>
                    ${escapeHTML(questionText)}
                </h3>

            </div>


            <div class="answer-section">

                <label for="student-answer">
                    Your Answer
                </label>

                <textarea
                    id="student-answer"
                    rows="5"
                    placeholder="Type your answer here..."
                ></textarea>

            </div>


            <button
                id="submit-assessment"
                onclick="submitCurrentAssessment()"
            >
                Submit Answer
            </button>

        </div>
    `;

}


/* =========================================================
   SUBMIT ANSWER
========================================================= */

async function submitCurrentAssessment() {

    if (!currentAssessment) {
        return;
    }

    const answerInput =
        document.getElementById("student-answer");

    if (!answerInput) {
        return;
    }

    const studentAnswer =
        answerInput.value.trim();

    if (!studentAnswer) {

        alert(
            "Please enter an answer first."
        );

        return;
    }


    const submitButton =
        document.getElementById(
            "submit-assessment"
        );

    if (submitButton) {

        submitButton.disabled = true;

        submitButton.textContent =
            "Analyzing...";

    }


    try {

        const result =
            await submitAssessmentAnswer(

                currentStudent.student_id,

                currentTopic,

                currentAssessment.question,

                currentAssessment.correct_answer,

                studentAnswer

            );


        console.log(
            "Assessment analysis:",
            result
        );


        renderAssessmentResult(result);

    } catch (error) {

        console.error(
            "Assessment submission failed:",
            error
        );

        alert(
            "Unable to analyze your answer."
        );


        if (submitButton) {

            submitButton.disabled = false;

            submitButton.textContent =
                "Submit Answer";

        }

    }

}


/* =========================================================
   RENDER RESULT
========================================================= */

function renderAssessmentResult(result) {

    const container =
        document.getElementById("assessment");

    if (!container) {
        return;
    }


    const isCorrect =
        Boolean(result.is_correct);


    const resultTitle =
        isCorrect
            ? "Correct!"
            : "Let's work on this.";


    const resultClass =
        isCorrect
            ? "correct-result"
            : "incorrect-result";


    /*
     * We don't assume the exact structure
     * of LearningAnalyst's response.
     *
     * The raw result is shown in a readable
     * fallback section as well.
     */

    const explanation =
        result.explanation ||
        result.feedback ||
        result.analysis ||
        "";


    const misconception =
        result.misconception ||
        result.misconception_label ||
        "";


    const nextAction =
        result.next_action ||
        result.action ||
        "";


    container.innerHTML = `

        <div class="card assessment-result">

            <div class="${resultClass}">

                <h2>
                    ${resultTitle}
                </h2>

            </div>


            <div class="result-section">

                <h3>Your Answer</h3>

                <p>
                    ${escapeHTML(
                        result.student_answer ||
                        "Submitted answer"
                    )}
                </p>

            </div>


            ${
                explanation
                ? `
                    <div class="result-section">

                        <h3>Analysis</h3>

                        <p>
                            ${escapeHTML(explanation)}
                        </p>

                    </div>
                `
                : ""
            }


            ${
                misconception
                ? `
                    <div class="result-section">

                        <h3>Detected Misconception</h3>

                        <p>
                            ${escapeHTML(misconception)}
                        </p>

                    </div>
                `
                : ""
            }


            ${
                nextAction
                ? `
                    <div class="result-section">

                        <h3>Recommended Next Step</h3>

                        <p>
                            ${escapeHTML(nextAction)}
                        </p>

                    </div>
                `
                : ""
            }


            <div class="assessment-actions">

                <button
                    onclick="startAssessment()"
                >
                    Next Question
                </button>

                <button
                    onclick="switchPage('tutor')"
                >
                    Ask AI Tutor
                </button>

            </div>


            <details class="raw-result">

                <summary>
                    View model analysis
                </summary>

                <pre>
${escapeHTML(
    JSON.stringify(result, null, 2)
)}
                </pre>

            </details>

        </div>
    `;

}


/* =========================================================
   CONNECT EXISTING ASSESSMENT BUTTON
========================================================= */

const assessmentButtonElement =
    document.getElementById("start-assessment");


if (assessmentButtonElement) {

    /*
     * Remove the old listener from app.js
     * by replacing the button behavior here.
     *
     * The actual button is recreated whenever
     * the assessment page is rendered.
     */

    assessmentButtonElement.onclick =
        startAssessment;

}