/* =========================================================
   QUINTET DASHBOARD
========================================================= */


/* =========================================================
   LOAD LEARNER PROFILE
========================================================= */

async function loadLearnerProfile() {

    if (!currentStudent) {
        console.warn("No student available.");
        return;
    }

    try {

        const profile = await getProfile(
            currentStudent.student_id,
            currentTopic
        );

        console.log(
            "Learner profile:",
            profile
        );

        updateDashboardProfile(profile);

    } catch (error) {

        console.error(
            "Could not load learner profile:",
            error
        );

    }

}


/* =========================================================
   UPDATE DASHBOARD
========================================================= */

function updateDashboardProfile(profile) {

    const levelElement =
        document.getElementById("learning-level");

    const recommendationElement =
        document.getElementById("recommendations");

    const recommendationList =
        document.getElementById("recommendation-list");


    /* -------------------------
       LEVEL
    ------------------------- */

    if (levelElement) {

        levelElement.textContent =
            profile.level || "--";

    }


    /* -------------------------
       KNOWN GAPS
    ------------------------- */

    const gaps =
        profile.known_gaps || [];


    if (recommendationElement) {

        if (gaps.length === 0) {

            recommendationElement.innerHTML = `
                <div class="recommendation-item">
                    No known learning gaps.
                    Keep practicing your current topic.
                </div>
            `;

        } else {

            recommendationElement.innerHTML =
                gaps
                    .slice(0, 5)
                    .map(gap => `
                        <div class="recommendation-item">
                            <strong>${escapeHTML(gap)}</strong>
                            <br>
                            <small>
                                Recommended prerequisite
                            </small>
                        </div>
                    `)
                    .join("");

        }

    }


    /* -------------------------
       RECOMMENDATION PAGE
    ------------------------- */

    if (recommendationList) {

        if (gaps.length === 0) {

            recommendationList.innerHTML = `
                <div class="recommendation-item">
                    You're currently on track.
                </div>
            `;

        } else {

            recommendationList.innerHTML =
                gaps
                    .map(gap => `
                        <div class="recommendation-item">
                            <strong>${escapeHTML(gap)}</strong>
                            <p>
                                Work on this prerequisite
                                before continuing.
                            </p>
                        </div>
                    `)
                    .join("");

        }

    }


    /* -------------------------
       MISCONCEPTIONS
    ------------------------- */

    updateMisconceptions(
        profile.misconception_log
    );

}


/* =========================================================
   MISCONCEPTIONS
========================================================= */

function updateMisconceptions(misconceptions) {

    if (!misconceptions) {
        return;
    }

    console.log(
        "Learner misconceptions:",
        misconceptions
    );

}


/* =========================================================
   NEXT ACTION
========================================================= */

async function loadNextAction() {

    if (!currentStudent) {
        return;
    }

    try {

        const result =
            await getNextAction(
                currentStudent.student_id,
                currentTopic
            );

        console.log(
            "Recommended next action:",
            result
        );

        return result;

    } catch (error) {

        console.error(
            "Could not determine next action:",
            error
        );

    }

}


/* =========================================================
   DASHBOARD INITIALIZATION
========================================================= */

async function initializeRealDashboard() {

    if (!currentStudent) {
        return;
    }

    await loadLearnerProfile();

    await loadNextAction();

}


/* =========================================================
   HTML SAFETY
========================================================= */

function escapeHTML(value) {

    if (value === null || value === undefined) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}