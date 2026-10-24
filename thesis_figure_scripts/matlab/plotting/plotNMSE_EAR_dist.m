% =========================================================================
% Script: plot_NMSE_AR_vs_Distance_Standalone_Styled.m
% Purpose: Standalone plotting script using exact data extracted from:
%          - NMSEvsDIST.fig
%          - ARvsDIST.fig
%          Updated to match the professional publication visual styling.
% =========================================================================
clear; clc; close all;

%% 1. Distance Vector
Dist_Vec = [0.4 0.6 0.8 1.0 1.2 1.4 1.6];

%% 2. Exact extracted data from NMSEvsDIST.fig
NMSE_OMP_dB     = [-6.7805824, -6.5222645, -7.1191616, -7.4148600, -7.6233720, -7.5804014, -7.6363444];
NMSE_SOMP_dB    = [-7.5571850, -7.3538446, -7.8813553, -8.3206280, -8.6984410, -8.4742620, -8.6977020];
NMSE_OMP_DR_dB  = [-7.8402130, -8.0512630, -8.3898980, -8.9598930, -9.1126790, -9.5012460, -9.3372530];
NMSE_SOMP_DR_dB = [-8.8294520, -9.0974200, -9.3814200, -10.1979300, -10.4889630, -11.0093140, -10.6766330];

%% 3. Exact extracted data from ARvsDIST.fig
AR_PCSI    = [39.762400, 36.795895, 34.127640, 31.990015, 30.327477, 29.036795, 28.017820];
AR_OMP     = [30.445704, 28.606339, 27.163118, 25.588524, 24.273836, 22.831917, 22.137306];
AR_SOMP    = [30.545218, 28.660694, 27.143623, 25.621233, 24.373072, 22.869957, 22.266426];
AR_OMP_DR  = [35.301685, 33.796036, 31.380241, 29.675625, 28.331990, 26.738945, 25.834352];
AR_SOMP_DR = [35.327568, 33.869873, 31.402681, 29.701990, 28.379282, 26.753180, 25.867195];

%% Visual Properties
lw = 2;       % Line width for algorithms
ms = 8;       % Marker size

%% ---------------------------------------------------------
% 4. Plot NMSE vs Distance
% ---------------------------------------------------------
figure('Color', [1,1,1], 'Name', 'NMSE vs Distance');
hold on; box on;

% OMP: Red Square
plot(Dist_Vec, NMSE_OMP_dB, '-s', 'Color', 'r', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r');

% SOMP: Green Diamond
plot(Dist_Vec, NMSE_SOMP_dB, '-d', 'Color', 'g', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'g');

% OMP-DR: Blue Asterisk
plot(Dist_Vec, NMSE_OMP_DR_dB, '-*', 'Color', 'b', 'LineWidth', lw, ...
    'MarkerSize', ms);

% SOMP-DR: Cyan Right-triangle
plot(Dist_Vec, NMSE_SOMP_DR_dB, '->', 'Color', 'c', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'c');

% Formatting
xlabel('Distance (m)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('NMSE [dB]', 'FontSize', 14, 'Interpreter', 'latex');


grid on; grid minor;
ax = gca;
ax.FontSize = 13;
ax.TickLabelInterpreter = 'latex';
ax.GridAlpha = 0.4;
ax.MinorGridAlpha = 0.5;
ax.MinorGridLineStyle = ':'; 
xlim([0.4 1.6]);

legend('OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', ...
    'Location', 'southwest', 'Interpreter', 'latex', 'FontSize', 12);

%% ---------------------------------------------------------
% 5. Plot Achievable Rate vs Distance
% ---------------------------------------------------------
figure('Color', [1,1,1], 'Name', 'Achievable Rate vs Distance');
hold on; box on;

% Perfect CSI: Black Solid Line
plot(Dist_Vec, AR_PCSI, '-k', 'LineWidth', 2.5);

% OMP: Red Square
plot(Dist_Vec, AR_OMP, '-s', 'Color', 'r', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'r');

% SOMP: Green Diamond
plot(Dist_Vec, AR_SOMP, '-d', 'Color', 'g', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'g');

% OMP-DR: Blue Asterisk
plot(Dist_Vec, AR_OMP_DR, '-*', 'Color', 'b', 'LineWidth', lw, ...
    'MarkerSize', ms);

% SOMP-DR: Cyan Right-triangle
plot(Dist_Vec, AR_SOMP_DR, '->', 'Color', 'c', 'LineWidth', lw, ...
    'MarkerSize', ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'c');

% Formatting
xlabel('Distance (m)', 'FontSize', 14, 'Interpreter', 'latex');
ylabel('Achievable Rate [bits/sec/Hz]', 'FontSize', 14, 'Interpreter', 'latex');


grid on; grid minor;
ax = gca;
ax.FontSize = 13;
ax.TickLabelInterpreter = 'latex';
ax.GridAlpha = 0.4;
ax.MinorGridAlpha = 0.5;
ax.MinorGridLineStyle = ':'; 
xlim([0.4 1.6]);

legend('Perfect CSI', 'OMP', 'SOMP', 'OMP-DR', 'SOMP-DR', ...
    'Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 12);