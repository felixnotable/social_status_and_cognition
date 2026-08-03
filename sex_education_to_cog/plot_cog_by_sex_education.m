%% Plot mean Cog changes by sex for four education groups
% This script reads:
%   HRS_Cog_mean_by_sex_education_w8_w13.csv
%
% It creates four separate line graphs:
%   1. Education < 12 years
%   2. Education = 12 years
%   3. Education = 13-15 years
%   4. Education >= 16 years
%
% Place this .m file and the CSV file in the same folder before running.

clear;
clc;
close all;

%% File settings
inputFile = "HRS_Cog_mean_by_sex_education_w8_w13.csv";
outputFolder = "cog_graphs";

if ~isfile(inputFile)
    error("Input file not found: %s", inputFile);
end

if ~isfolder(outputFolder)
    mkdir(outputFolder);
end

%% Read the mean-value table
data = readtable(inputFile, "TextType", "string");

requiredVariables = [
    "sex"
    "education_group"
    "mean_Cog_wave8"
    "mean_Cog_wave9"
    "mean_Cog_wave10"
    "mean_Cog_wave11"
    "mean_Cog_wave12"
    "mean_Cog_wave13"
];

missingVariables = setdiff(requiredVariables, string(data.Properties.VariableNames));

if ~isempty(missingVariables)
    error( ...
        "The input CSV is missing required variables: %s", ...
        strjoin(missingVariables, ", ") ...
    );
end

%% Graph definitions
waves = 8:13;

educationGroups = [
    "Less than 12 years"
    "12 years"
    "13-15 years"
    "16 or more years"
];

graphTitles = [
    "Mean Cog Change by Sex: Education < 12 Years"
    "Mean Cog Change by Sex: Education = 12 Years"
    "Mean Cog Change by Sex: Education = 13-15 Years"
    "Mean Cog Change by Sex: Education >= 16 Years"
];

outputFiles = [
    "cog_change_education_lt12_by_sex.png"
    "cog_change_education_12_by_sex.png"
    "cog_change_education_13_15_by_sex.png"
    "cog_change_education_16plus_by_sex.png"
];

cogVariables = [
    "mean_Cog_wave8"
    "mean_Cog_wave9"
    "mean_Cog_wave10"
    "mean_Cog_wave11"
    "mean_Cog_wave12"
    "mean_Cog_wave13"
];

%% Create one graph for each education group
for groupIndex = 1:numel(educationGroups)

    groupName = educationGroups(groupIndex);

    maleRow = data( ...
        data.sex == "Male" & data.education_group == groupName, ...
        : ...
    );

    femaleRow = data( ...
        data.sex == "Female" & data.education_group == groupName, ...
        : ...
    );

    if height(maleRow) ~= 1
        error( ...
            "Expected exactly one male row for education group '%s'; found %d.", ...
            groupName, ...
            height(maleRow) ...
        );
    end

    if height(femaleRow) ~= 1
        error( ...
            "Expected exactly one female row for education group '%s'; found %d.", ...
            groupName, ...
            height(femaleRow) ...
        );
    end

    maleCog = zeros(1, numel(waves));
    femaleCog = zeros(1, numel(waves));

    for waveIndex = 1:numel(waves)
        variableName = cogVariables(waveIndex);
        maleCog(waveIndex) = maleRow.(variableName);
        femaleCog(waveIndex) = femaleRow.(variableName);
    end

    figure("Color", "white", "Position", [100, 100, 900, 560]);

    plot( ...
        waves, ...
        maleCog, ...
        "-o", ...
        "LineWidth", 1.8, ...
        "MarkerSize", 7, ...
        "DisplayName", "Male" ...
    );

    hold on;

    plot( ...
        waves, ...
        femaleCog, ...
        "-o", ...
        "LineWidth", 1.8, ...
        "MarkerSize", 7, ...
        "DisplayName", "Female" ...
    );

    hold off;

    xlabel("Wave");
    ylabel("Mean Cog");
    title(graphTitles(groupIndex));
    xticks(waves);
    xlim([8, 13]);
    grid on;
    legend("Location", "best");

    outputPath = fullfile(outputFolder, outputFiles(groupIndex));
    exportgraphics(gcf, outputPath, "Resolution", 300);

    fprintf("Created: %s\n", outputPath);
end

fprintf("\nAll four graphs were created in the folder: %s\n", outputFolder);
