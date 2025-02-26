use crate::client::gemini::{request, Stats};
use crate::translate::translator::Context;
use futures::{stream, StreamExt};
use log::{debug, error, trace};
use serde::Deserialize;

#[derive(Deserialize)]
struct Translated {
    text: Vec<String>,
}

pub struct BulkTranslated {
    pub number: i32,
    pub original_lines: Vec<String>,
    pub translated_lines: Vec<String>,
    pub stats: Stats,
}

pub async fn translate(context: &Context, lines: Vec<String>) -> Vec<String> {
    debug!("line_length:{}", lines.len());
    if lines.is_empty() {
        return lines;
    }
    translate_parallel(
        &context.language,
        &context.model,
        &context.api_key,
        lines,
        context.lines,
        context.requests,
        0,
    )
    .await
}

async fn translate_parallel(
    language: &String,
    model: &String,
    api_key: &String,
    lines: Vec<String>,
    chunk_lines: usize,
    requests: usize,
    retry_count: i32,
) -> Vec<String> {
    let mut number = 0;
    let bodies = stream::iter(lines.chunks(chunk_lines))
        .map(|chunked| {
            let language = language.clone();
            let model = model.clone();
            let api_key = api_key.clone();
            number += 1;
            let order_number = number;
            async move {
                translate_bulk(order_number, &language, &model, &api_key, chunked.to_vec()).await
            }
        })
        .buffer_unordered(requests);

    let mut responses = vec![];
    let mut bodies_stream = bodies;
    while let Some(response) = bodies_stream.next().await {
        responses.push(response);
    }

    responses.sort_by(|a, b| a.number.cmp(&b.number));
    let mut translated = vec![];
    for response in responses {
        let mut translated_lines = response.translated_lines;
        let original_lines = response.original_lines;
        response.stats.log();
        if translated_lines.len() != original_lines.len() {
            if retry_count > 4 {
                panic!("retry max error");
            }
            for l in &original_lines {
                trace!("{}", l);
            }
            for l in &translated_lines {
                trace!("{}", l);
            }
            error!("retry count: {}", retry_count);
            error!(
                "translated line length error {}/{}",
                translated_lines.len(),
                original_lines.len()
            );
            translated_lines = Box::pin(translate_parallel(
                language,
                model,
                api_key,
                original_lines,
                1,
                requests,
                retry_count + 1,
            ))
            .await;
        }
        translated.append(&mut translated_lines);
    }
    translated
}

async fn translate_bulk(
    number: i32,
    language: &String,
    model: &str,
    api_key: &String,
    original_lines: Vec<String>,
) -> BulkTranslated {
    let mut user_contents: Vec<String> = vec![];
    for line in &original_lines {
        user_contents.push(format!("<paragraph>{}</paragraph>", line));
    }

    let prompt = format!(
        "You are an expert translator of fantasy literature, proficient in multiple languages including Vietnamese and Han-Viet (Sino-Vietnamese), with a deep understanding of East Asian storytelling styles. I am providing you with a text segment from the novel 'Omniscient Reader’s Viewpoint' (Vietnamese title: 'Toàn trí độc giả'), a renowned Korean fantasy work translated into English. Your task is to translate this text into {} with the highest quality, adhering to the following requirements and rules:\n\
        1. Preserve the original storytelling style—vivid, humorous, and tense—as it appears in the source text.\n\
        2. If the target language is Vietnamese, use Han-Viet vocabulary for skill names, Constellation titles, and key concepts to create a formal, captivating tone that resonates with East Asian fantasy aesthetics. Specifically for Vietnamese:\n\
        - Translate 'Secretive Plotter' as 'Kẻ Mưu Phản Bí Mật'.\n\
        - Translate 'Black Flame Dragon' as 'Hắc Hỏa Vực Long'.\n\
        - Translate 'Prisoner of the Golden Headband' as 'Chủ Nhân của Vòng Kim Cô', and apply a similar style to other Constellation titles.\n\
        - Translate general terms as follows: 'Streamer' to 'Kẻ Phát Thanh', 'Scenario' to 'Kịch Bản', 'Incarnation' to 'Hóa Thân'.\n\
        3. Ensure no English or Chinese words remain in the translation—convert everything into the target language (except proper names like 'Kim Dokja' or 'Yoo Joonghyuk').\n\
        4. Produce a natural, fluent, and engaging translation that appeals to readers of fantasy literature in the target language.\n\
        Additional Translation Rules:\n\
        - Maintain semantic accuracy: Do not alter the meaning or intent of the original text.\n\
        - Avoid unnecessary repetition: Use varied vocabulary where appropriate to enhance readability, but keep key terms consistent.\n\
        - Prioritize consistency: Apply the same translation for recurring names, skills, or concepts throughout the text.\n\
        - Adapt idioms or cultural references: Localize them into equivalents that fit the fantasy context of the target language.\n\
        - Enhance tone where needed: Amplify the dramatic or emotional impact using expressive phrasing suited to the target language (e.g., Han-Viet for Vietnamese).\n\
        Translate it into {}. Please output the following JSON.\n\
        A string in `<paragraph>` tag to `</paragraph>` tag is one paragraph.\n\
        If a paragraph of input is translated and consists of multiple sentences, output an array consisting of multiple Strings.\n\
        There are {} paragraphs of input, please output {} lines.\n\
        Using this JSON schema:\n\
        Paragraph = {{\"line\": number, \"text\": list[string]}}\n\
        Return a `list[Paragraph]`.\n\
        Please remove `<paragraph>` and `</paragraph>` tags from the translation result.\n\
        Here is the text to translate:\n{}", 
        language, 
        language, 
        &original_lines.len(), 
        &original_lines.len(), 
        &original_lines.join("\n")
    );

    let response = request(model, api_key, &prompt, &user_contents)
        .await
        .expect("Gemini API Request Error");
    let translated_vec = serde_json::from_str::<Vec<Translated>>(response.text.trim());
    if translated_vec.is_err() {
        error!("JSON Parse error choice:{}", &response.text.trim());
        return BulkTranslated {
            number,
            original_lines,
            translated_lines: vec![],
            stats: response.stats,
        };
    }
    let mut translated_lines = vec![];
    for result in translated_vec.unwrap() {
        translated_lines.push(result.text.join("\n"));
    }

    BulkTranslated {
        number,
        original_lines,
        translated_lines,
        stats: response.stats,
    }
}
