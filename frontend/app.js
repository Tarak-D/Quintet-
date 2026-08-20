/* =========================================================
   QUINTET FRONTEND APPLICATION
========================================================= */


/* =========================================================
   GLOBAL STATE
========================================================= */

let currentStudent = null;
let currentTopic = "Mathematics";


/* =========================================================
   DOM ELEMENTS
========================================================= */

const navItems = document.querySelectorAll(".nav-item");
const pages = document.querySelectorAll(".page");
const pageTitle = document.getElementById("page-title");


/* =========================================================
   PAGE TITLES
========================================================= */

const pageTitles = {

    dashboard: "Dashboard",

    assessment: "Assessment",

    diagnostic: "Learning Diagnostic",

    tutor: "AI Tutor",

    recommendations: "Recommendations"

};


/* =========================================================
   NAVIGATION
========================================================= */

navItems.forEach(button => {

    button.addEventListener("click", () => {

        const page = button.dataset.page;

        switchPage(page);

    });

});


function switchPage(pageName) {

    /*
     * Update navigation buttons
     */

    navItems.forEach(button => {

        button.classList.remove("active");

        if (button.dataset.page === pageName) {
            button.classList.add("active");
        }

    });


    /*
     * Hide all pages
     */

    pages.forEach(page => {

        page.classList.remove("active");

    });


    /*
     * Recommendations uses a slightly
     * different ID in index.html
     */

    let targetPage = document.getElementById(pageName);

    if (pageName === "recommendations") {
        targetPage =
            document.getElementById("recommendations-page");
    }


    if (targetPage) {
        targetPage.classList.add("active");
    }


    /*
     * Update page title
     */

    if (pageTitle) {

        pageTitle.textContent =
            pageTitles[pageName] || "Quintet";

    }

}


/* =========================================================
   BACKEND CONNECTION
========================================================= */

async function initializeBackend() {

    try {

        const result = await checkBackendHealth();

        console.log("Quintet backend:", result);

    } catch (error) {

        console.error(
            "Unable to connect to Quintet backend."
        );

        showBackendStatus(false);

    }

}


function showBackendStatus(isOnline) {

    const studentName =
        document.getElementById("student-name");

    if (!studentName) {
        return;
    }

    if (isOnline) {

        studentName.textContent =
            "Backend Online";

    } else {

        studentName.textContent =
            "Backend Offline";

    }

}


/* =========================================================
   STUDENT CREATION
========================================================= */

async function initializeStudent() {

    /*
     * For now we use a simple demo student.
     *
     * Later this will become a proper login/
     * student-selection system.
     */

    const savedStudent =
        localStorage.getItem("quintet_student");


    if (savedStudent) {

        try {

            currentStudent =
                JSON.parse(savedStudent);

            updateStudentDisplay();

            return;

        } catch (error) {

            console.warn(
                "Invalid saved student. Creating a new one."
            );

            localStorage.removeItem(
                "quintet_student"
            );

        }

    }


    try {

        currentStudent =
            await createStudent("Demo Student");

        localStorage.setItem(
            "quintet_student",
            JSON.stringify(currentStudent)
        );

        updateStudentDisplay();

    } catch (error) {

        console.error(
            "Could not create student:",
            error
        );

        const studentName =
            document.getElementById("student-name");

        if (studentName) {

            studentName.textContent =
                "Demo Student";

        }

    }

}


/* =========================================================
   STUDENT DISPLAY
========================================================= */

function updateStudentDisplay() {

    const studentName =
        document.getElementById("student-name");

    if (!studentName || !currentStudent) {
        return;
    }

    studentName.textContent =
        currentStudent.display_name;

}


/* =========================================================
   DASHBOARD
========================================================= */

function initializeDashboard() {

    const accuracy =
        document.getElementById("overall-accuracy");

    const questions =
        document.getElementById("questions-attempted");

    const subjects =
        document.getElementById("subjects-count");

    const level =
        document.getElementById("learning-level");


    /*
     * Placeholder values.
     *
     * These will be replaced with real
     * learner-model data from the backend.
     */

    if (accuracy) {
        accuracy.textContent = "--";
    }

    if (questions) {
        questions.textContent = "--";
    }

    if (subjects) {
        subjects.textContent = "--";
    }

    if (level) {
        level.textContent = "--";
    }

}


/* =========================================================
   ASSESSMENT BUTTON
========================================================= */

const assessmentButton =
    document.getElementById("start-assessment");


if (assessmentButton) {

    assessmentButton.addEventListener(
        "click",
        async () => {

            if (!currentStudent) {

                alert(
                    "Student profile is not ready yet."
                );

                return;

            }


            try {

                const result =
                    await getAssessmentQuestion(
                        currentStudent.student_id,
                        currentTopic
                    );

                console.log(
                    "Assessment question:",
                    result
                );

                alert(
                    "Assessment question received. " +
                    "Check the browser console for now."
                );

            } catch (error) {

                alert(
                    "Unable to generate assessment question."
                );

            }

        }
    );

}


/* =========================================================
   DIAGNOSTIC BUTTON
========================================================= */

const diagnosticButton =
    document.getElementById("run-diagnostic");


if (diagnosticButton) {

    diagnosticButton.addEventListener(
        "click",
        async () => {

            const results =
                document.getElementById(
                    "diagnostic-results"
                );


            if (!results) {
                return;
            }


            results.textContent =
                "Generating diagnostic...";


            try {

                const diagnostic =
                    await startDiagnostic(
                        currentTopic
                    );


                console.log(
                    "Diagnostic:",
                    diagnostic
                );


                results.innerHTML = `
                    <p>
                        Diagnostic generated successfully.
                    </p>

                    <p>
                        Questions:
                        ${diagnostic.questions?.length || 0}
                    </p>
                `;

            } catch (error) {

                console.error(error);

                results.innerHTML = `
                    <p>
                        Unable to generate diagnostic.
                    </p>
                `;

            }

        }
    );

}


/* =========================================================
   AI TUTOR
========================================================= */

const askTutorButton =
    document.getElementById("ask-tutor");

const questionInput =
    document.getElementById("question-input");

const chatBox =
    document.getElementById("chat-box");


if (askTutorButton) {

    askTutorButton.addEventListener(
        "click",
        askTutorQuestion
    );

}


if (questionInput) {

    questionInput.addEventListener(
        "keydown",
        event => {

            if (event.key === "Enter") {

                askTutorQuestion();

            }

        }
    );

}


async function askTutorQuestion() {

    if (!questionInput || !chatBox) {
        return;
    }


    const question =
        questionInput.value.trim();


    if (!question) {
        return;
    }


    /*
     * Show student's message
     */

    const userMessage =
        document.createElement("div");

    userMessage.className =
        "user-message";

    userMessage.textContent =
        question;

    chatBox.appendChild(
        userMessage
    );


    questionInput.value = "";


    /*
     * Temporary loading message
     */

    const loadingMessage =
        document.createElement("div");

    loadingMessage.className =
        "bot-message";

    loadingMessage.textContent =
        "Thinking...";

    chatBox.appendChild(
        loadingMessage
    );


    chatBox.scrollTop =
        chatBox.scrollHeight;


    try {

        if (!currentStudent) {
            throw new Error(
                "Student not initialized."
            );
        }


        /*
         * The existing backend's /tutor/teach
         * endpoint expects a topic rather than
         * a raw question.
         *
         * We'll connect the full conversational
         * flow once we inspect orchestrator.py.
         */

        const response =
            await teach(
                currentStudent.student_id,
                currentTopic,
                true,
                question
            );


        loadingMessage.textContent =
            extractTutorResponse(response);


    } catch (error) {

        console.error(error);

        loadingMessage.textContent =
            "Sorry, I couldn't reach the AI tutor.";

    }


    chatBox.scrollTop =
        chatBox.scrollHeight;

}


/* =========================================================
   TUTOR RESPONSE HELPER
========================================================= */

function extractTutorResponse(response) {

    if (!response) {
        return "No response received.";
    }


    /*
     * Try common response fields.
     */

    if (typeof response === "string") {
        return response;
    }


    if (response.response) {
        return response.response;
    }


    if (response.explanation) {
        return response.explanation;
    }


    if (response.message) {
        return response.message;
    }


    if (response.answer) {
        return response.answer;
    }


    /*
     * Fallback for unknown backend structure.
     */

    return JSON.stringify(response);

}


/* =========================================================
   INITIALIZATION
========================================================= */

async function initializeQuintet() {

    console.log(
        "Initializing Quintet frontend..."
    );


    initializeDashboard();

    await initializeBackend();

    await initializeStudent();

    await initializeRealDashboard();


    console.log(
        "Quintet frontend initialized."
    );

}


/* =========================================================
   START APPLICATION
========================================================= */

document.addEventListener(
    "DOMContentLoaded",
    initializeQuintet
);