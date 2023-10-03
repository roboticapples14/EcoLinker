# EcoLinker
EcoLinker is a tool to determine the most cost-effective allocation of restoration resources for maximizing habitat connectivity. This package serves as a more practical interface for Government + conservation agencies + decision makers who want to utilize EcoScape and other connectivity models to inform real conservation decisions. The program takes as input the habitat, connectivity and flow layers for a set of birds, a budget (dollar amount) for restoration, terrain, and a terrain conversion cost map (the dollar amount of converting one type of terrain to another). Its objective is to find the optimal terrain conversion for maximizing habitat connectivity.

# Problem statement
In the face of unexpected environmental changes caused by climate change and encroaching human development, many species are reliant on deliberate human intervention to ensure the connectivity and health of their habitat. Many nonprofit organizations exist today whose main objective is to protect and restore land that's ecologically significant in an effort to increase biodiversity and preserve habitat health. Through methods like erosion control, reforestation and revegetation[1] they can restore land to a suitable state for the many species that rely on it. However, these efforts are costly and are sometimes even ineffective, due to the extremely multivariate nature of complex ecosystems. Given that restoration efforts only have finite resources and money, it's crucial to use it as effectively as possible in order to reach their goals and environmental deliverables. With the effects of climate change advancing every year, it is one of the most important questions of our time how to achieve these environmental goals such as "reversing habitat and species loss" with such limited resources.

Nonprofits and environmental managers rely heavily on ecological modeling to inform their conservation decisions. While there are many species distribution models widely used today for these conservation efforts, many are inefficient and therefore extremely time-consuming, costly, and even environmentally unfriendly to run. Additionally, most are quite technical, due to the fact that the tools are made primarily for ecologists and scientists, and not for business/logistic decision making. This adds a layer of inaccessibility to the tolls, requiring an intermediary of professional scientific analysis to translate a connectivity map into a restoration strategy.

# Solution
With EcoScape, we've dramatically improved the time it takes to run these models for bird distribution simulations. However, the inaccessability is still a massive blocker for non-scientists to leverage the power of these efficient simulations. The goal of this project is to bridge the gap between EcoScape's powerful connectivity simulation and environmental resource managers who need tools to accurately inform restoration decisions to be cost-effective and high-impact.

# Project Objective:
	- Input: 
		○ Species habitat
		○ Terrain geotiff
		○ Dollar amount available for restoration (or num of pixels to convert)
		○ N x N Matrix representing cost required to convert terrain of type i to terrain type j
		○ (potentially connectivity + flow layers)
		○ (potentially type of terrain to restore to)
		
	- Output: 
		○ Optimal terrain conversion (for maximizing habitat connectivity)
			- Format: ??? Geotiff with just new terrain? Or CSV with recommendation terrain conversion, location, cost
