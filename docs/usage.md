## Main Panel Page

![main_interface](https://raw.githubusercontent.com/Yanjieij/IPCC_kotaemon/refs/heads/main/docs/images/main_interface.PNG)

The main interface of IPCC RAG system is divided into 3 regions:

1. Left: **Conversation Settings Panel**
   - Here you can select, create, rename, and delete conversations.
   - Below that you have the file index, where you can choose whether to disable, select all files, or select which files to retrieve references from.
     - If you choose "Search All", all files will be considered during chat.
     - If you choose "Select", a dropdown will appear for you to select the
       files to be considered during chat. If no files are selected, then no
       files will be considered during chat.
2. Middle: **Chat Panel**
   - This is where you can chat with the chatbot.
3. Right: **Information Panel**
   - This is the region that visualization features and context information will be displayed.

## Quick Start

1. Log in by entering the guest account username and password.

2. Access the main interface, click the `green pencil` icon at the left region to create a new conversation.

   ![new_conversation](https://raw.githubusercontent.com/Yanjieij/IPCC_kotaemon/refs/heads/main/docs/images/new_conversation.PNG)

3. Select the RAG strategy from the feature blocks at the left region. Note that only the three options starting with `IPCC` should be used.

   ![select_strategy](https://raw.githubusercontent.com/Yanjieij/IPCC_kotaemon/refs/heads/main/docs/images/select_strategy.PNG)

4. Use `Search in Files` to select the chapter (since the report is very long, `Searching ALL` will be extremely slow).

   ![select_chapters](https://raw.githubusercontent.com/Yanjieij/IPCC_kotaemon/refs/heads/main/docs/images/select_chapters.PNG)

5. Input your question regarding the selected chapter in the middle conversation region and click send.

6. Wait for the response, currently each Q&A might take about 30 seconds.

7. When `Conversation Renamed` appears at the top right, this indicates that the current round of conversation is completed, and you can take further actions.

   ![conversation_renamed](https://raw.githubusercontent.com/Yanjieij/IPCC_kotaemon/refs/heads/main/docs/images/conversation_renamed.PNG)

### Notes

- The system needs some time to load database and conversation history, so if you found the files or conversations empty, please `wait for a while`.
- The `time required` to generate response is not fixed. In general, the more chapters or strategies you selected, the longer it will take.
- After you input the question, please `wait` for the `Conversation Renamed` to appear before performing any other operations.

## Chat Settings

Currently, the IPCC RAG system can be used immediately after login, no additional system settings are required.

However, there are still some feature settings:

### RAG Strategies

- **IPCC Traditional RAG** (Traditional RAG with Plain Text)

  The fastest strategy, suitable only for answering questions related to the textual content.

- **IPCC GraphRAG** (GraphRAG)

  Slow, suitable for answering summary-type or abstract-type questions.

- **IPCC PDF Traditional RAG** (Traditional RAG with Figures)

  Relatively fast, allows for the retrieval of figures and tables within the documents.

### Chat Settings

In the chat region, you can click the grey triangle to expand the **setting panel**.

However, some features here are not prepared, what you can modify are the `Mindmap` and `Chat suggestion`:

You could select whether the mindmap and chat suggestion is needed or not. Other options are **not ready** now.

![chat_settings](https://raw.githubusercontent.com/Yanjieij/IPCC_kotaemon/refs/heads/main/docs/images/chat_settings.PNG)

## Reference Questions

The following questions can serve as a reference for users to help them formulate queries related to the IPCC reports.

### Chapter 1：Point of Departure and Key Concepts

- What does the "Point of Departure and Key Concepts" in the IPCC report specifically refer to?
- Why is it important to consider vulnerability in climate risk assessments?
- How is disaster risk management connected to climate change adaptation?
- What is meant by "transformation," and what role does it play in meeting societal goals?
- In this report, what aspects are covered under "lived experiences"?
- How can individual extreme weather events be attributed to greenhouse gas emissions?

### Chapter 2：Terrestrial and Freshwater Ecosystems and Their Services

- What are ecosystem services, and what examples of these services are mentioned in the chapter?
- How do terrestrial and freshwater ecosystems contribute to carbon storage, and why is this important for climate change mitigation?
- What does the term "novel fire regimes" refer to, and how can they impact grassy shrublands as mentioned in the Nama-Karoo region?
- How might increased thermal stress affect plant communities in desert ecosystems, such as the Sonoran Desert?
- What risks do Mediterranean-Type Ecosystems (MTEs) face due to projected changes in temperature and precipitation patterns?
- In what ways do wildfires and tree mortality following such events impact water quality and availability?

### Chapter 3：Oceans and Coastal Ecosystems and Their Services

- What are the specific mechanisms through which climate change exacerbates the effects of non-climate anthropogenic drivers on marine life?
- How do Earth system models (ESMs) contribute to our understanding of the long-term impacts of climate change on marine ecosystems?
- What are the main challenges in detecting and attributing changes in marine and coastal ecosystems to specific climatic or non-climatic drivers?
- How do the interactions between multiple stressors (e.g., warming, acidification, and deoxygenation) affect the resilience of marine ecosystems?
- What are the potential consequences of reduced biodiversity in equatorial and Arctic marine regions, even under moderate warming scenarios?
- How do ecosystem-based adaptation (EbA) measures compare to more traditional engineering solutions in terms of effectiveness and sustainability?

### Chapter 4：Water

- How are cultural water uses of Indigenous Peoples, local communities, and traditional peoples being affected by climate change?
- What are some examples of adaptation options for agriculture to cope with changing water availability?
- Why is there a concern about the potential impact of bioenergy with carbon capture and storage (BECCS) on water resources?
- How might the COVID-19 pandemic influence the financing of water adaptation measures?
- In what ways does gender, equity, and social justice relate to water access and climate change adaptation?
- What are the projected changes in heavy precipitation, and how do they differ from changes in mean precipitation?

### Chapter 5：Food, Fibre and Other Ecosystem Products

- What are the main ways in which climate change is expected to affect global food security?
- How does climate change impact the safety of our food, and what kind of contaminants are we more likely to see?
- Can you explain what RCP4.5 and RCP8.5 scenarios mean in the context of future food security?
- What are some of the key challenges faced by small-scale food producers due to climate change?
- How will climate change affect the availability of micronutrients in our diets, and why is this important for health?
- What adaptation options are available for agriculture, fisheries, and forestry, and why is there a need for increased investment in these areas?

### Chapter 6：Cities, Settlements and Key Infrastructure

- How does urbanization contribute to climate risk, and what can be done to reduce it?
- Can you explain what "informal settlements" are and why they are particularly vulnerable to climate impacts?
- What role do social movements play in addressing climate change in urban areas, and how have they evolved since the last IPCC report?
- How does inclusive action, such as gender-responsive and intersectional approaches, help in adapting to climate change?
- What are some examples of co-benefits that come from implementing sustainable development in cities?
- Why is it important to consider the health effects of urban heat islands and heatwaves in the context of climate change?

### Chapter 7：Health, Wellbeing and the Changing Structure of Communities

- Can you explain the connection between climate change and involuntary population displacements?
- In what ways can adaptation and sustainable development contribute to peace in regions prone to conflict?
- What role do natural resource management and gender-based approaches play in achieving sustainable peace?
- How is the term "health and well-being" interpreted within the context of this chapter?
- What are the key differences between the impacts of climate change on urban and rural populations?
- How do changes in the cryosphere and ocean systems affect human health and well-being?

### Chapter 8：Poverty, Livelihoods and Sustainable Development

- What are some examples of livelihood strategies used by communities in atoll regions facing displacement?
- How does the COVID-19 pandemic relate to the issues discussed in this chapter?
- In what ways can conflict and governance influence the response to climate change?
- What are the key factors that contribute to human vulnerability in the face of climate change?
- How do extreme poverty and inequality increase the risk of adverse impacts from climate change?
- What role does access to healthcare and basic infrastructure play in a community's resilience to climate change?

### Chapter 9：Africa

- What are the observed impacts of climate change on staple crops in sub-Saharan Africa, and which crops have been most affected?
- How has climate change specifically impacted maize and wheat yields in sub-Saharan Africa, and what are the percentages of decrease?
- What is the overall reduction in total food calories across all crops in sub-Saharan Africa due to climate change since 1970, and which countries have experienced the highest reductions?
- What are some of the adaptation options listed for preventing malnutrition, and how do they relate to urban agriculture and forestry?
- How can building materials be used to improve the ability of housing to moderate indoor temperatures, and why is this important for health?
- In what ways can cash transfers assist vulnerable groups in absorbing the negative impacts of climate-related shocks or stress, and what is a 'Cash+' approach?

### Chapter 10：Asia

- How does the perception of climate risk influence individual adaptation behaviors in Asian communities?
- Can you explain the concept of "perceived self-efficacy" and its role in the context of climate change adaptation?
- What are some innovative water management solutions, such as artificial glaciers and ice stupas, and how do they work in the Himalayan regions?
- In what ways can traditional knowledge systems complement scientific approaches to address climate change vulnerabilities in Asia?
- How might the projected sea level rise affect the major deltas in Asia, and which populations are most at risk?
- What are the key challenges and options for adaptation in the freshwater sector, and why is an integrated management approach important?

### Chapter 11：Australasia

- How do economic activities in Australia and New Zealand contribute to their vulnerability or resilience to climate impacts?
- What are the key risks associated with changes in the structure and composition of ecosystems in Australia and New Zealand?
- How do demographic and social trends influence the exposure, vulnerability, and adaptive capacity of communities in Australasia?
- What are the projected impacts of climate change on inequalities between Indigenous and non-Indigenous peoples in the region?
- What are the main sectors of the economy in Australia and New Zealand, and how might they be affected by climate change?
- What are some of the governance frameworks needed to facilitate the transition to more climate-resilient development pathways?

### Chapter 12：Central and South America

- What are the projected changes in biomes across South America by the year 2100, and why is this significant?
- How are climate-induced changes different from land use changes, and why is it challenging to distinguish between them?
- In what ways are human health in CSA being affected by changes in weather and climatic patterns?
- What are the risks associated with the increase in wildfires in South America, and how do these risks relate to climate change?
- How might the geographic range and seasonality of malaria transmission be altered by climate change, and what are the implications for public health?
- What are the benefits of limiting global warming to 1.5°C for ecosystems and communities in Central America and the Amazon?

### Chapter 13：Europe

- Can you explain the difference between vulnerability and exposure to climate impacts, as discussed in the document?
- How might a 2°C global warming level (GWL) affect water scarcity and desertification in Europe?
- What are some of the key factors driving sensitivity to desertification in certain regions of Europe?
- Why are mountain ecosystems and glaciers particularly sensitive to heat, and what are the implications of their loss?
- What are the residual risks that may occur even with adaptation measures in place, and why are they significant?
- How do the adaptation plans of cities and settlements differ across Europe, and what are the challenges faced by smaller or economically weaker cities?

### Chapter 14：North America

- What are the key differences between incremental and transformational adaptation approaches as described in the table?
- Could you provide an example of how urban areas can adapt to extreme storms and flooding according to the document?
- What does the term "polar amplification" refer to, and why is it significant for North America?
- How have average temperatures in North America changed since 1960, and which regions have seen the most pronounced warming?
- What are some of the observed changes in North American climate, such as reduced snowpack and sea ice extent?
- Why is there a focus on the Arctic region within this chapter, and how is it relevant to the rest of North America?

### Chapter 15：Small Islands

- How do small islands' geographical characteristics contribute to their vulnerability to climate change?
- What are the Sustainable Development Goals (SDGs), and why are they important for small islands?
- In what ways can education and awareness-raising help communities adapt to climate change?
- What are some examples of adaptation measures that have been implemented in small islands?
- How does the integration of Indigenous Knowledge and Local Knowledge (IKLK) benefit small islands in preparing for and recovering from tropical cyclones?
- What role do social capital and health-related strategies play in the resilience of small island communities?

### Chapter 16：Key Risks across Sectors and Regions

- What is meant by "limits to adaptation," and why is it an important concept in climate change discussions?
- How do the projected impacts of a 2°C global temperature increase compare with a 1.5°C increase?
- What are some of the observed impacts of climate change, and how have communities responded to these changes?
- What are the main challenges in aligning climate change adaptation with the Sustainable Development Goals?
- Could you provide examples of how gender considerations are important for successful adaptation efforts?
- How does the timing of risk accrual affect the level of risk associated with climate change?

### Chapter 17：Decision-Making Options for Managing Risk

- Can you explain what "practitioner knowledge" means, and how it differs from local knowledge?
- How does the integration of different governance portfolios help in managing climate risks?
- What role do early-warning systems play in managing risks associated with climate change?
- Why is it important to consider decision-making procedures alongside the options for responding to climate change?
- What are "no-regret options," and why are they significant in the context of climate risk management?
- How does the urgency of the climate crisis affect the way we make decisions about adaptation?

### Chapter 18：Climate Resilient Development Pathways

- Can you explain what "fuel switching" means in the context of energy systems transition?
- What kind of financial incentives are being discussed for promoting renewable energy?
- How can urban planning and infrastructure be improved to increase resilience against climate change?
- What role do women farmers play in enhancing food security and climate resilience?
- Why is it important to strengthen land tenure security and access to land for climate adaptation?
- How do payments for ecosystem services work, and how do they help with climate resilience?